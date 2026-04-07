"""
Qwen3-TTS ファインチューニングスクリプト

公式 finetuning/sft_12hz.py をベースに以下のバグ修正を適用:
1. ラベルシフトの二重適用修正 (Issue #179)
2. sub-talkerの損失関数修正 (PR #278)
3. text_projection欠落修正 (PR #188)

使い方:
  python finetune.py prepare --input_dir <音声セグメントディレクトリ> --train_jsonl <出力先>
  python finetune.py train --train_jsonl <学習データ> --output_dir <出力先> [オプション]
"""

import argparse
import json
import logging
import os
import shutil
import sys

import torch

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


# --- データ準備 ---

def prepare_data(args: argparse.Namespace) -> None:
    """音声セグメントからaudio_codesを抽出し、学習用JSONLを生成する。"""
    from qwen_tts import Qwen3TTSTokenizer

    logger.info(f"Tokenizerをロード中: {args.tokenizer_model}")
    tokenizer = Qwen3TTSTokenizer.from_pretrained(
        args.tokenizer_model,
        device_map=args.device,
    )

    # 入力JSONLを読み込み
    with open(args.input_jsonl, "r", encoding="utf-8") as f:
        lines = [json.loads(line.strip()) for line in f if line.strip()]

    logger.info(f"{len(lines)}サンプルのaudio_codesを抽出します")

    # バッチ処理
    batch_size = 32
    final_lines = []
    for i in range(0, len(lines), batch_size):
        batch = lines[i:i + batch_size]
        batch_audios = [item["audio"] for item in batch]

        enc_res = tokenizer.encode(batch_audios)
        for code, item in zip(enc_res.audio_codes, batch):
            item["audio_codes"] = code.cpu().tolist()
            final_lines.append(item)

        logger.info(f"  {min(i + batch_size, len(lines))}/{len(lines)} 完了")

    with open(args.output_jsonl, "w", encoding="utf-8") as f:
        for item in final_lines:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    logger.info(f"学習データを保存しました: {args.output_jsonl}（{len(final_lines)}サンプル）")


# --- 学習 ---

def train(args: argparse.Namespace) -> None:
    """バグ修正済みのファインチューニングを実行する。"""
    import torch.nn.functional as F
    from accelerate import Accelerator
    from qwen_tts.inference.qwen3_tts_model import Qwen3TTSModel
    from safetensors.torch import save_file
    from torch.optim import AdamW
    from torch.utils.data import DataLoader
    from transformers import AutoConfig

    # dataset.pyの内容をインラインで定義（公式と同等）
    from finetune_dataset import TTSDataset

    accelerator = Accelerator(
        gradient_accumulation_steps=args.grad_accum,
        mixed_precision="bf16",
    )

    logger.info(f"モデルをロード中: {args.init_model}")
    # FlashAttention2が使えない場合はsdpaにフォールバック
    try:
        qwen3tts = Qwen3TTSModel.from_pretrained(
            args.init_model,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
        )
        logger.info("FlashAttention2を使用します")
    except Exception:
        qwen3tts = Qwen3TTSModel.from_pretrained(
            args.init_model,
            torch_dtype=torch.bfloat16,
            attn_implementation="sdpa",
        )
        logger.info("SDPAにフォールバックしました")

    config = AutoConfig.from_pretrained(args.init_model)

    with open(args.train_jsonl, "r", encoding="utf-8") as f:
        train_data = [json.loads(line) for line in f if line.strip()]

    dataset = TTSDataset(train_data, qwen3tts.processor, config)
    train_dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=dataset.collate_fn,
    )

    optimizer = AdamW(qwen3tts.model.parameters(), lr=args.lr, weight_decay=0.01)

    model, optimizer, train_dataloader = accelerator.prepare(
        qwen3tts.model, optimizer, train_dataloader
    )

    target_speaker_embedding = None
    model.train()

    logger.info(f"学習開始: {args.num_epochs}エポック, batch_size={args.batch_size}, lr={args.lr}")

    for epoch in range(args.num_epochs):
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(model):
                input_ids = batch["input_ids"]
                codec_ids = batch["codec_ids"]
                ref_mels = batch["ref_mels"]
                text_embedding_mask = batch["text_embedding_mask"]
                codec_embedding_mask = batch["codec_embedding_mask"]
                attention_mask = batch["attention_mask"]
                codec_0_labels = batch["codec_0_labels"]
                codec_mask = batch["codec_mask"]

                speaker_embedding = model.speaker_encoder(
                    ref_mels.to(model.device).to(model.dtype)
                ).detach()
                if target_speaker_embedding is None:
                    target_speaker_embedding = speaker_embedding

                input_text_ids = input_ids[:, :, 0]
                input_codec_ids = input_ids[:, :, 1]

                input_text_embedding = model.talker.model.text_embedding(input_text_ids)

                # [修正3] text_projection対応（0.6Bモデル対応）
                if hasattr(model.talker, "text_projection"):
                    input_text_embedding = model.talker.text_projection(input_text_embedding)

                input_text_embedding = input_text_embedding * text_embedding_mask
                input_codec_embedding = model.talker.model.codec_embedding(input_codec_ids) * codec_embedding_mask
                input_codec_embedding[:, 6, :] = speaker_embedding

                input_embeddings = input_text_embedding + input_codec_embedding

                for i in range(1, 16):
                    codec_i_embedding = model.talker.code_predictor.get_input_embeddings()[i - 1](
                        codec_ids[:, :, i]
                    )
                    codec_i_embedding = codec_i_embedding * codec_mask.unsqueeze(-1)
                    input_embeddings = input_embeddings + codec_i_embedding

                # [修正1] ラベルシフトの二重適用を修正
                # HuggingFaceのForCausalLMLossが内部でシフトするため、手動シフトを削除
                outputs = model.talker(
                    inputs_embeds=input_embeddings,
                    attention_mask=attention_mask,
                    labels=codec_0_labels,
                    output_hidden_states=True,
                )

                # sub-talker loss: 正しいマスク適用
                hidden_states = outputs.hidden_states[0][-1]
                target_codec_mask = codec_mask[:, 1:]
                talker_hidden_states = hidden_states[:, :-1][target_codec_mask]
                talker_codec_ids = codec_ids[:, 1:][target_codec_mask]

                # [修正2] sub-talkerの損失関数をcross_entropyに直接差し替え
                sub_talker_loss = torch.tensor(0.0, device=model.device)
                if talker_hidden_states.numel() > 0:
                    sub_logits_list = []
                    sub_labels_list = []
                    for i in range(1, 16):
                        predictor = model.talker.code_predictor.predictors[i - 1]
                        logits_i = predictor(talker_hidden_states)
                        sub_logits_list.append(logits_i)
                        sub_labels_list.append(talker_codec_ids[:, i])

                    sub_logits = torch.cat(sub_logits_list, dim=0)
                    sub_labels = torch.cat(sub_labels_list, dim=0)
                    sub_talker_loss = F.cross_entropy(sub_logits, sub_labels, ignore_index=-100)

                loss = outputs.loss + 0.3 * sub_talker_loss

                accelerator.backward(loss)

                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), 1.0)

                optimizer.step()
                optimizer.zero_grad()

            if step % 10 == 0:
                accelerator.print(
                    f"エポック {epoch + 1}/{args.num_epochs} | ステップ {step} | Loss: {loss.item():.4f}"
                )

        # チェックポイント保存
        if accelerator.is_main_process:
            output_dir = os.path.join(args.output_dir, f"checkpoint-epoch-{epoch}")
            shutil.copytree(args.init_model, output_dir, dirs_exist_ok=True)

            # config.jsonを更新
            config_file = os.path.join(output_dir, "config.json")
            with open(config_file, "r", encoding="utf-8") as f:
                config_dict = json.load(f)

            config_dict["tts_model_type"] = "custom_voice"
            talker_config = config_dict.get("talker_config", {})
            talker_config["spk_id"] = {args.speaker_name: 3000}
            talker_config["spk_is_dialect"] = {args.speaker_name: False}
            config_dict["talker_config"] = talker_config

            with open(config_file, "w", encoding="utf-8") as f:
                json.dump(config_dict, f, indent=2, ensure_ascii=False)

            # モデル重みを保存
            unwrapped_model = accelerator.unwrap_model(model)
            state_dict = {
                k: v.detach().to("cpu") for k, v in unwrapped_model.state_dict().items()
            }

            # speaker_encoderの重みは不要
            keys_to_drop = [k for k in state_dict if k.startswith("speaker_encoder")]
            for k in keys_to_drop:
                del state_dict[k]

            # ターゲット話者のembeddingをcodec_embeddingに書き込み
            weight = state_dict["talker.model.codec_embedding.weight"]
            state_dict["talker.model.codec_embedding.weight"][3000] = (
                target_speaker_embedding[0].detach().to(weight.device).to(weight.dtype)
            )
            save_file(state_dict, os.path.join(output_dir, "model.safetensors"))

            logger.info(f"チェックポイントを保存しました: {output_dir}")

    logger.info("学習完了")


def main():
    parser = argparse.ArgumentParser(description="Qwen3-TTS ファインチューニング")
    subparsers = parser.add_subparsers(dest="command")

    # prepare サブコマンド
    p_prepare = subparsers.add_parser("prepare", help="学習データの前処理")
    p_prepare.add_argument("--device", type=str, default="cuda:0")
    p_prepare.add_argument(
        "--tokenizer_model", type=str, default="Qwen/Qwen3-TTS-Tokenizer-12Hz"
    )
    p_prepare.add_argument("--input_jsonl", type=str, required=True)
    p_prepare.add_argument("--output_jsonl", type=str, required=True)

    # train サブコマンド
    p_train = subparsers.add_parser("train", help="ファインチューニング実行")
    p_train.add_argument(
        "--init_model", type=str, default="Qwen/Qwen3-TTS-12Hz-1.7B-Base"
    )
    p_train.add_argument("--output_dir", type=str, default="ft_output")
    p_train.add_argument("--train_jsonl", type=str, required=True)
    p_train.add_argument("--batch_size", type=int, default=4)
    p_train.add_argument("--lr", type=float, default=2e-5)
    p_train.add_argument("--num_epochs", type=int, default=5)
    p_train.add_argument("--grad_accum", type=int, default=4)
    p_train.add_argument("--speaker_name", type=str, default="custom_speaker")

    args = parser.parse_args()

    if args.command == "prepare":
        prepare_data(args)
    elif args.command == "train":
        train(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
