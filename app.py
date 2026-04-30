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

# 設定ファイル読み込み
with open("config.yaml", "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)

# エンジン管理: Qwen3-TTS と GPT-SoVITS を遅延ロードし、
# UI 上のラジオボタンで切替可能にする。
ENGINE_CHOICES = ["qwen3_tts", "gpt_sovits"]
ENGINE_LABELS = {"qwen3_tts": "Qwen3-TTS", "gpt_sovits": "GPT-SoVITS"}

engines: dict[str, object] = {}
engine = None  # 現在アクティブなエンジン
current_engine_name: str = config.get("engine", "qwen3_tts")
current_prompt: dict | None = None
current_prompt_name: str | None = None


def _load_engine(name: str):
    """指定エンジンを遅延ロードして engines dict にキャッシュ。"""
    if name in engines:
        return engines[name]
    if name == "qwen3_tts":
        logger.info("Qwen3-TTS モジュールを読み込み中（初回は時間がかかります）...")
        from clone_engine import VoiceCloneEngine
        engines[name] = VoiceCloneEngine()
    elif name == "gpt_sovits":
        logger.info("GPT-SoVITS モジュールを読み込み中...")
        from sovits_engine import SovitsEngine
        engines[name] = SovitsEngine()
    else:
        raise ValueError(f"未対応のエンジン: {name}")
    logger.info(f"エンジン {name} のロード完了")
    return engines[name]


def _switch_engine(name: str):
    """アクティブエンジンを切り替える。グローバル engine を更新。"""
    global engine, current_engine_name, current_prompt, current_prompt_name
    engine = _load_engine(name)
    current_engine_name = name
    current_prompt = None
    current_prompt_name = None


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


def on_engine_change(engine_name: str) -> tuple[gr.update, str]:
    """エンジン切替: アクティブエンジンを変更し、保存済み一覧を更新する。"""
    _switch_engine(engine_name)
    label = ENGINE_LABELS.get(engine_name, engine_name)
    return (
        gr.update(choices=engine.list_prompts(), value=None),
        f"エンジンを「{label}」に切り替えました。",
    )


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


def on_ft_upload(audio_files: list[str] | None, script_text: str) -> tuple[str, str]:
    """FT用音声アップロード: 分割 + 原稿テキスト対応付け"""
    if not audio_files:
        raise gr.Error("音声ファイルをアップロードしてください。")
    if not script_text.strip():
        raise gr.Error("原稿テキストを入力してください。")

    # 原稿をパース
    script_lines = audio_utils.parse_script(script_text)
    if not script_lines:
        raise gr.Error("原稿テキストから文を抽出できませんでした。")

    ft_data_dir = config["paths"].get("ft_data_dir", "ft_data")
    segments_dir = os.path.join(ft_data_dir, "segments")
    # 既存セグメントをクリア
    if os.path.exists(segments_dir):
        import shutil
        shutil.rmtree(segments_dir)
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

        # 無音区間で分割（原稿の文数に合わせて閾値を自動調整）
        segments = audio_utils.split_audio_by_silence(
            tmp_wav, segments_dir, num_expected=len(script_lines),
        )
        all_segments.extend(segments)

    # 原稿テキストを対応付け
    all_segments = audio_utils.assign_script_to_segments(all_segments, script_lines)

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
    n_match = "一致" if len(all_segments) == len(script_lines) else f"不一致（原稿{len(script_lines)}文）"
    summary = f"分割数: {len(all_segments)}セグメント（{n_match}）\n"
    summary += f"参照音声: {os.path.basename(ref_audio)}\n\n"
    for i, seg in enumerate(all_segments):
        summary += f"[{i + 1}] ({seg['duration']:.1f}秒) {seg['text']}\n"

    status = f"{len(all_segments)}セグメントに分割完了。データ: {jsonl_path}"
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
    if not hasattr(engine, "load_ft_model"):
        raise gr.Error("ファインチューニングは Qwen3-TTS でのみ利用できます。エンジンを切り替えてください。")
    engine.load_ft_model(checkpoint_name)
    return f"FTモデル「{checkpoint_name}」をロードしました。"


def on_ft_generate(
    text: str, speaker_name: str, language: str,
) -> tuple[str | None, str | None]:
    """FTモデルで音声生成する。"""
    if not hasattr(engine, "generate_ft_long"):
        raise gr.Error("ファインチューニングは Qwen3-TTS でのみ利用できます。エンジンを切り替えてください。")
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


def on_ft_load_script_template() -> str:
    """録音用原稿テンプレートを読み込む。"""
    script_path = "docs/finetune_script.md"
    if not os.path.exists(script_path):
        raise gr.Error("原稿テンプレートが見つかりません。")

    # Markdownから番号付きリスト部分だけ抽出
    import re
    lines = []
    with open(script_path, "r", encoding="utf-8") as f:
        for line in f:
            match = re.match(r"^\d+\.\s+(.+)", line.strip())
            if match:
                lines.append(line.strip())

    return "\n".join(lines)


CUSTOM_CSS = """
@import url('https://fonts.googleapis.com/css2?family=EB+Garamond:ital,wght@0,400;0,600;1,400&display=swap');
* { font-family: 'EB Garamond', 'Times New Roman', Times, serif, 'Noto Sans JP', sans-serif !important; }
textarea, input, select, button { font-family: 'EB Garamond', 'Times New Roman', Times, serif, 'Noto Sans JP', sans-serif !important; }
"""


def build_ui() -> gr.Blocks:
    """Gradio UIを構築する。"""
    ft_config = config.get("finetune", {})

    with gr.Blocks(title="Voice Clone TTS", theme=gr.themes.Soft(), css=CUSTOM_CSS) as demo:
        gr.Markdown("# Voice Clone TTS\n自分の声をアップロードして、任意のテキストを音読させよう")

        # === エンジン切替（共通ヘッダ） ===
        with gr.Row():
            engine_select = gr.Radio(
                choices=[(ENGINE_LABELS[k], k) for k in ENGINE_CHOICES],
                value=current_engine_name,
                label="TTSエンジン",
                interactive=True,
            )
            engine_status = gr.Textbox(
                label="エンジンステータス",
                value=f"現在のエンジン: {ENGINE_LABELS.get(current_engine_name, current_engine_name)}",
                interactive=False,
            )

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
                    gr.Markdown("#### 1. 原稿を用意して録音")
                    gr.Markdown(
                        "原稿を読み上げた録音と、その原稿テキストをセットでアップロードしてください。\n"
                        "各文の間に1秒程度の間を空けて読むと、自動分割の精度が上がります。"
                    )
                    ft_script_load_btn = gr.Button("録音用原稿テンプレートを読み込む")
                    ft_script_input = gr.Textbox(
                        label="原稿テキスト（1行1文、または「1. テキスト」形式）",
                        lines=10,
                        placeholder="1. 今日はとても良い天気ですね。\n2. 朝早く起きて、近くの公園を散歩してきました。\n...",
                    )
                    ft_audio_input = gr.File(
                        label="録音ファイル（複数可、MP3/WAV/M4A）",
                        file_count="multiple",
                        file_types=["audio"],
                    )
                    ft_upload_btn = gr.Button("音声を分割・原稿を対応付け", variant="primary")
                    ft_upload_status = gr.Textbox(label="ステータス", interactive=False)
                    ft_segments_preview = gr.Textbox(
                        label="セグメント一覧（音声と原稿の対応）",
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
                        choices=(
                            engine.list_ft_models()
                            if engine and hasattr(engine, "list_ft_models")
                            else []
                        ),
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

        # エンジン切替
        engine_select.change(
            fn=on_engine_change,
            inputs=[engine_select],
            outputs=[saved_prompts, engine_status],
        )

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
        ft_script_load_btn.click(
            fn=on_ft_load_script_template,
            outputs=[ft_script_input],
        )
        ft_upload_btn.click(
            fn=on_ft_upload,
            inputs=[ft_audio_input, ft_script_input],
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
            fn=lambda: gr.update(
                choices=engine.list_ft_models() if hasattr(engine, "list_ft_models") else []
            ),
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
    # config.yaml の engine 値で初期エンジンをロード
    _switch_engine(current_engine_name)
    demo = build_ui()
    demo.launch(
        server_name=config["server"]["host"],
        server_port=config["server"]["port"],
    )
