"""Gradio Web UI（メインエントリポイント）"""

import datetime
import json
import logging
import os
import subprocess
import tempfile
import threading

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

logger.info("ライブラリを読み込み中...")
import torch
import gradio as gr
import soundfile as sf
import yaml

# CUDA状態をログ出力
if torch.cuda.is_available():
    logger.info(f"CUDA: 有効 (GPU: {torch.cuda.get_device_name(0)}, VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB)")
else:
    logger.warning(f"CUDA: 無効 (PyTorch版: {torch.__version__}, CUDA版でない場合はGPUが使えません)")

import audio_utils
logger.info("Qwen3-TTSモジュールを読み込み中（初回は時間がかかります）...")
from clone_engine import VoiceCloneEngine
logger.info("読み込み完了")

# 設定ファイル読み込み
with open("config.yaml", "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)

# グローバルにエンジンを保持（起動時にモデルロード）
engine: VoiceCloneEngine | None = None
current_prompt: dict | None = None
current_prompt_name: str | None = None


def on_upload(audio_file: str | None) -> str:
    """音声アップロード時の処理。変換・バリデーション・書き起こしを行う。"""
    if audio_file is None:
        return ""

    # WAVに変換
    tmp_wav = tempfile.mktemp(suffix=".wav")
    try:
        audio_utils.convert_to_wav(audio_file, tmp_wav)
    except RuntimeError as e:
        raise gr.Error(str(e))

    # バリデーション
    is_valid, message = audio_utils.validate_audio(tmp_wav)
    if not is_valid:
        raise gr.Error(message)
    if message.startswith("警告"):
        gr.Warning(message)

    # 書き起こし
    try:
        transcript = audio_utils.transcribe_audio(tmp_wav)
    except Exception as e:
        raise gr.Error(f"書き起こしに失敗しました: {e}")

    return transcript


def on_register(
    audio_file: str | None,
    transcript: str,
    prompt_name: str,
) -> tuple[str, gr.update, str | None]:
    """声の登録処理。promptを生成・保存する。"""
    global current_prompt, current_prompt_name

    if audio_file is None:
        raise gr.Error("音声ファイルをアップロードしてください。")
    if not prompt_name.strip():
        raise gr.Error("声の名前を入力してください。")

    # WAVに変換
    tmp_wav = tempfile.mktemp(suffix=".wav")
    audio_utils.convert_to_wav(audio_file, tmp_wav)

    # バリデーション
    is_valid, message = audio_utils.validate_audio(tmp_wav)
    if not is_valid:
        raise gr.Error(message)

    # Prompt生成
    ref_text = transcript.strip() if transcript.strip() else None
    current_prompt = engine.create_voice_prompt(tmp_wav, ref_text)
    current_prompt_name = prompt_name.strip()

    # 保存
    engine.save_prompt(current_prompt, prompt_name.strip())

    # スペクトログラム生成
    ref_spec = audio_utils.generate_spectrogram(tmp_wav, "参照音声スペクトログラム")

    # ドロップダウン更新
    choices = engine.list_prompts()
    return (
        f"「{prompt_name.strip()}」として登録しました。",
        gr.update(choices=choices, value=prompt_name.strip()),
        ref_spec,
    )


def on_generate(text: str, language: str, dl_format: str) -> tuple[str | None, str | None, str | None, str | None]:
    """音声生成処理。"""
    if current_prompt is None:
        raise gr.Error("先に声を登録するか、保存済みの声を読み込んでください。")
    if not text.strip():
        raise gr.Error("テキストを入力してください。")

    # 音声生成
    audio_array, sample_rate = engine.generate_long(text.strip(), current_prompt, language)

    # ファイル保存（WAV）
    output_dir = config["paths"]["output_dir"]
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_text = text.strip()[:10].replace("/", "_").replace("\\", "_")
    wav_path = os.path.join(output_dir, f"{timestamp}_{safe_text}.wav")
    sf.write(wav_path, audio_array, sample_rate)

    # MP3に変換
    mp3_path = wav_path.replace(".wav", ".mp3")
    audio_utils.convert_wav_to_mp3(wav_path, mp3_path)

    # スペクトログラム生成
    gen_spec = audio_utils.generate_spectrogram(wav_path, "生成音声スペクトログラム")

    # 選択された形式のファイルをダウンロードに設定
    dl_path = mp3_path if dl_format == "MP3" else wav_path

    # 音声ラベルに声の名前を表示
    label = f"生成結果 ― {current_prompt_name} から生成" if current_prompt_name else "生成結果"

    logger.info(f"音声を保存しました: {wav_path}, {mp3_path}")
    return gr.update(value=wav_path, label=label), dl_path, wav_path, gen_spec


def on_switch_format(dl_format: str, wav_state: str | None) -> str | None:
    """ダウンロード形式の切り替え。"""
    if wav_state is None:
        return None
    if dl_format == "MP3":
        mp3_path = wav_state.replace(".wav", ".mp3")
        if os.path.exists(mp3_path):
            return mp3_path
    return wav_state


def on_load_prompt(prompt_name: str | None) -> str:
    """保存済みpromptの読み込み。"""
    global current_prompt, current_prompt_name

    if not prompt_name:
        raise gr.Error("読み込む声を選択してください。")

    current_prompt = engine.load_prompt(prompt_name)
    current_prompt_name = prompt_name
    return f"「{prompt_name}」を読み込みました。"


def on_delete_prompt(prompt_name: str | None) -> tuple[str, gr.update]:
    """保存済みpromptの削除。"""
    global current_prompt

    if not prompt_name:
        raise gr.Error("削除する声を選択してください。")

    engine.delete_prompt(prompt_name)
    current_prompt = None
    choices = engine.list_prompts()
    return (
        f"「{prompt_name}」を削除しました。",
        gr.update(choices=choices, value=None),
    )


# --- ファインチューニング関連 ---

ft_training_log: list[str] = []
ft_is_training = False


def on_ft_upload(audio_files: list[str] | None) -> tuple[str, str]:
    """FT用音声アップロード: 分割 + 書き起こし"""
    if not audio_files:
        raise gr.Error("音声ファイルをアップロードしてください。")

    ft_data_dir = config["paths"].get("ft_data_dir", "ft_data")
    segments_dir = os.path.join(ft_data_dir, "segments")
    os.makedirs(segments_dir, exist_ok=True)

    all_segments = []
    for audio_file in audio_files:
        # WAV 24kHzに変換（FTデータ用）
        tmp_wav = tempfile.mktemp(suffix=".wav")
        cmd = [
            "ffmpeg", "-y", "-i", audio_file,
            "-ar", "24000", "-ac", "1", "-sample_fmt", "s16", tmp_wav,
        ]
        subprocess.run(cmd, capture_output=True, check=True)

        # 無音区間で分割
        segments = audio_utils.split_audio_by_silence(tmp_wav, segments_dir)
        all_segments.extend(segments)

    # 書き起こし
    all_segments = audio_utils.transcribe_segments(all_segments)

    # 参照音声: 最初のセグメントを使用
    ref_audio = all_segments[0]["path"]

    # JSONL生成
    jsonl_path = os.path.join(ft_data_dir, "train_raw.jsonl")
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for seg in all_segments:
            item = {
                "audio": seg["path"],
                "text": seg["text"],
                "ref_audio": ref_audio,
            }
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    # 確認用テキスト
    summary = f"分割数: {len(all_segments)}セグメント\n参照音声: {os.path.basename(ref_audio)}\n\n"
    for i, seg in enumerate(all_segments):
        summary += f"[{i + 1}] ({seg['duration']:.1f}秒) {seg['text']}\n"

    status = f"{len(all_segments)}セグメントに分割・書き起こし完了。データ: {jsonl_path}"
    return status, summary


def on_ft_start(
    speaker_name: str,
    num_epochs: int,
    batch_size: int,
    learning_rate: float,
) -> str:
    """ファインチューニングを開始する。"""
    global ft_is_training, ft_training_log

    if ft_is_training:
        raise gr.Error("学習が既に実行中です。")

    ft_data_dir = config["paths"].get("ft_data_dir", "ft_data")
    raw_jsonl = os.path.join(ft_data_dir, "train_raw.jsonl")
    if not os.path.exists(raw_jsonl):
        raise gr.Error("学習データが準備されていません。先に音声をアップロードしてください。")

    if not speaker_name.strip():
        raise gr.Error("スピーカー名を入力してください。")

    ft_training_log = []
    ft_is_training = True

    def run_training():
        global ft_is_training
        try:
            prepared_jsonl = os.path.join(ft_data_dir, "train_with_codes.jsonl")
            ft_output_dir = config["paths"].get("ft_output_dir", "ft_output")

            # Step 1: データ前処理
            ft_training_log.append("=== データ前処理開始 ===")
            cmd_prepare = [
                "uv", "run", "python", "finetune.py", "prepare",
                "--input_jsonl", raw_jsonl,
                "--output_jsonl", prepared_jsonl,
                "--device", "cuda:0",
            ]
            proc = subprocess.run(cmd_prepare, capture_output=True, text=True)
            ft_training_log.append(proc.stdout)
            if proc.returncode != 0:
                ft_training_log.append(f"エラー: {proc.stderr}")
                return

            # Step 2: 学習実行
            ft_training_log.append("=== 学習開始 ===")
            cmd_train = [
                "uv", "run", "python", "finetune.py", "train",
                "--train_jsonl", prepared_jsonl,
                "--output_dir", ft_output_dir,
                "--speaker_name", speaker_name.strip(),
                "--num_epochs", str(num_epochs),
                "--batch_size", str(batch_size),
                "--lr", str(learning_rate),
            ]
            proc = subprocess.Popen(
                cmd_train, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, bufsize=1,
            )
            for line in proc.stdout:
                ft_training_log.append(line.rstrip())
            proc.wait()

            if proc.returncode == 0:
                ft_training_log.append("=== 学習完了 ===")
            else:
                ft_training_log.append(f"=== 学習失敗 (exit code: {proc.returncode}) ===")
        except Exception as e:
            ft_training_log.append(f"エラー: {e}")
        finally:
            ft_is_training = False

    thread = threading.Thread(target=run_training, daemon=True)
    thread.start()

    return "学習を開始しました。ログを確認してください。"


def on_ft_get_log() -> str:
    """学習ログを取得する。"""
    if not ft_training_log:
        return "ログはまだありません。"
    status = "【学習中...】\n" if ft_is_training else "【完了】\n"
    return status + "\n".join(ft_training_log[-100:])


def on_ft_load_model(checkpoint_name: str | None) -> str:
    """FTモデルをロードする。"""
    if not checkpoint_name:
        raise gr.Error("チェックポイントを選択してください。")
    engine.load_ft_model(checkpoint_name)
    return f"FTモデル「{checkpoint_name}」をロードしました。"


def on_ft_generate(
    text: str, speaker_name: str, language: str,
) -> tuple[str | None, str | None]:
    """FTモデルで音声生成する。"""
    if not hasattr(engine, "_ft_checkpoint"):
        raise gr.Error("先にFTモデルをロードしてください。")
    if not text.strip():
        raise gr.Error("テキストを入力してください。")

    audio_array, sample_rate = engine.generate_ft_long(
        text.strip(), speaker_name.strip(), language,
    )

    output_dir = config["paths"]["output_dir"]
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_text = text.strip()[:10].replace("/", "_").replace("\\", "_")
    wav_path = os.path.join(output_dir, f"ft_{timestamp}_{safe_text}.wav")
    sf.write(wav_path, audio_array, sample_rate)

    mp3_path = wav_path.replace(".wav", ".mp3")
    audio_utils.convert_wav_to_mp3(wav_path, mp3_path)

    gen_spec = audio_utils.generate_spectrogram(wav_path, "FT生成音声スペクトログラム")

    logger.info(f"FT音声を保存しました: {wav_path}")
    return wav_path, mp3_path, gen_spec


def build_ui() -> gr.Blocks:
    """Gradio UIを構築する。"""
    custom_css = """
    @import url('https://fonts.googleapis.com/css2?family=EB+Garamond:ital,wght@0,400;0,600;1,400&display=swap');
    * { font-family: 'EB Garamond', 'Times New Roman', Times, serif, 'Noto Sans JP', sans-serif !important; }
    textarea, input, select, button { font-family: 'EB Garamond', 'Times New Roman', Times, serif, 'Noto Sans JP', sans-serif !important; }
    """
    ft_config = config.get("finetune", {})

    with gr.Blocks(title="Voice Clone TTS", theme=gr.themes.Soft(), css=custom_css) as demo:
        gr.Markdown("# Voice Clone TTS\n自分の声をアップロードして、任意のテキストを音読させよう")

        with gr.Tabs():
            # === ボイスクローンタブ（既存） ===
            with gr.Tab("ボイスクローン"):
                with gr.Group():
                    gr.Markdown("### Step 1: 声をアップロード")
                    audio_input = gr.Audio(
                        sources=["upload"],
                        type="filepath",
                        label="音声ファイル (MP3/WAV/M4A, 3~30秒)",
                    )
                    transcript_box = gr.Textbox(
                        label="書き起こし結果（自動入力・編集可）",
                        lines=2,
                    )
                    prompt_name_input = gr.Textbox(
                        label="声の名前",
                        value="my_voice",
                        max_lines=1,
                    )
                    register_btn = gr.Button("声を登録", variant="primary")
                    register_status = gr.Textbox(label="ステータス", interactive=False)
                    ref_spectrogram = gr.Image(label="参照音声スペクトログラム", visible=True)

                with gr.Group():
                    gr.Markdown("### Step 2: 声を選択")
                    saved_prompts = gr.Dropdown(
                        label="保存済みの声を選択",
                        choices=engine.list_prompts() if engine else [],
                    )
                    load_btn = gr.Button("読み込む")
                    delete_btn = gr.Button("削除", variant="stop")
                    action_status = gr.Textbox(label="ステータス", interactive=False)

                with gr.Group():
                    gr.Markdown("### Step 3: テキストを入力")
                    text_input = gr.Textbox(
                        label="読み上げたいテキスト",
                        lines=5,
                        placeholder="ここにテキストを入力...",
                    )
                    language_select = gr.Dropdown(
                        choices=["Japanese", "English", "Chinese", "Korean"],
                        value="Japanese",
                        label="言語",
                    )
                    generate_btn = gr.Button("音声を生成", variant="primary")
                    audio_output = gr.Audio(label="生成結果", type="filepath")
                    with gr.Row():
                        dl_format = gr.Radio(
                            choices=["WAV", "MP3"],
                            value="WAV",
                            label="ダウンロード形式",
                        )
                        dl_file = gr.File(label="ダウンロード")
                    wav_state = gr.State(value=None)
                    gen_spectrogram = gr.Image(label="生成音声スペクトログラム", visible=True)

            # === ファインチューニングタブ ===
            with gr.Tab("ファインチューニング"):
                gr.Markdown(
                    "### ファインチューニング\n"
                    "長尺の音声データを使ってモデルを学習し、より忠実な音声再現を実現します。"
                )

                with gr.Group():
                    gr.Markdown("#### 1. 学習用音声のアップロード")
                    ft_audio_input = gr.File(
                        label="音声ファイル（複数可、MP3/WAV/M4A）",
                        file_count="multiple",
                        file_types=["audio"],
                    )
                    ft_upload_btn = gr.Button("音声を分割・書き起こし", variant="primary")
                    ft_upload_status = gr.Textbox(label="ステータス", interactive=False)
                    ft_segments_preview = gr.Textbox(
                        label="セグメント一覧（書き起こし結果）",
                        lines=10,
                        interactive=False,
                    )

                with gr.Group():
                    gr.Markdown("#### 2. 学習パラメータ")
                    ft_speaker_name = gr.Textbox(
                        label="スピーカー名",
                        value="custom_speaker",
                        max_lines=1,
                    )
                    with gr.Row():
                        ft_epochs = gr.Number(
                            label="エポック数",
                            value=ft_config.get("default_epochs", 5),
                            precision=0,
                        )
                        ft_batch_size = gr.Number(
                            label="バッチサイズ",
                            value=ft_config.get("default_batch_size", 4),
                            precision=0,
                        )
                        ft_lr = gr.Number(
                            label="学習率",
                            value=ft_config.get("default_lr", 2e-5),
                        )
                    ft_start_btn = gr.Button("学習開始", variant="primary")
                    ft_start_status = gr.Textbox(label="ステータス", interactive=False)

                with gr.Group():
                    gr.Markdown("#### 3. 学習ログ")
                    ft_log_btn = gr.Button("ログを更新")
                    ft_log_output = gr.Textbox(
                        label="学習ログ",
                        lines=15,
                        interactive=False,
                    )

                with gr.Group():
                    gr.Markdown("#### 4. FTモデルで音声生成")
                    ft_model_select = gr.Dropdown(
                        label="チェックポイントを選択",
                        choices=engine.list_ft_models() if engine else [],
                    )
                    ft_model_refresh_btn = gr.Button("一覧を更新")
                    ft_load_btn = gr.Button("モデルをロード", variant="primary")
                    ft_load_status = gr.Textbox(label="ステータス", interactive=False)

                    ft_text_input = gr.Textbox(
                        label="読み上げたいテキスト",
                        lines=5,
                        placeholder="ここにテキストを入力...",
                    )
                    ft_speaker_for_gen = gr.Textbox(
                        label="スピーカー名（学習時と同じ名前）",
                        value="custom_speaker",
                        max_lines=1,
                    )
                    ft_language_select = gr.Dropdown(
                        choices=["Japanese", "English", "Chinese", "Korean"],
                        value="Japanese",
                        label="言語",
                    )
                    ft_generate_btn = gr.Button("音声を生成", variant="primary")
                    ft_audio_output = gr.Audio(label="FT生成結果", type="filepath")
                    ft_dl_file = gr.File(label="MP3ダウンロード")
                    ft_gen_spectrogram = gr.Image(label="FT生成音声スペクトログラム")

        # --- イベントバインディング ---

        # ボイスクローンタブ
        audio_input.change(
            fn=on_upload,
            inputs=[audio_input],
            outputs=[transcript_box],
        )
        register_btn.click(
            fn=on_register,
            inputs=[audio_input, transcript_box, prompt_name_input],
            outputs=[register_status, saved_prompts, ref_spectrogram],
        )
        generate_btn.click(
            fn=on_generate,
            inputs=[text_input, language_select, dl_format],
            outputs=[audio_output, dl_file, wav_state, gen_spectrogram],
        )
        dl_format.change(
            fn=on_switch_format,
            inputs=[dl_format, wav_state],
            outputs=[dl_file],
        )
        load_btn.click(
            fn=on_load_prompt,
            inputs=[saved_prompts],
            outputs=[action_status],
        )
        delete_btn.click(
            fn=on_delete_prompt,
            inputs=[saved_prompts],
            outputs=[action_status, saved_prompts],
        )

        # ファインチューニングタブ
        ft_upload_btn.click(
            fn=on_ft_upload,
            inputs=[ft_audio_input],
            outputs=[ft_upload_status, ft_segments_preview],
        )
        ft_start_btn.click(
            fn=on_ft_start,
            inputs=[ft_speaker_name, ft_epochs, ft_batch_size, ft_lr],
            outputs=[ft_start_status],
        )
        ft_log_btn.click(
            fn=on_ft_get_log,
            outputs=[ft_log_output],
        )
        ft_model_refresh_btn.click(
            fn=lambda: gr.update(choices=engine.list_ft_models()),
            outputs=[ft_model_select],
        )
        ft_load_btn.click(
            fn=on_ft_load_model,
            inputs=[ft_model_select],
            outputs=[ft_load_status],
        )
        ft_generate_btn.click(
            fn=on_ft_generate,
            inputs=[ft_text_input, ft_speaker_for_gen, ft_language_select],
            outputs=[ft_audio_output, ft_dl_file, ft_gen_spectrogram],
        )

    return demo


if __name__ == "__main__":
    engine = VoiceCloneEngine()
    demo = build_ui()
    demo.launch(
        server_name=config["server"]["host"],
        server_port=config["server"]["port"],
    )
