"""Gradio Web UI（メインエントリポイント）"""

import datetime
import logging
import os
import tempfile

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

logger.info("ライブラリを読み込み中...")
import gradio as gr
import soundfile as sf
import yaml

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
    global current_prompt

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


def on_generate(text: str, language: str) -> tuple[str | None, str | None, str | None]:
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

    logger.info(f"音声を保存しました: {wav_path}, {mp3_path}")
    return wav_path, gr.update(value=mp3_path, visible=True), gen_spec


def on_load_prompt(prompt_name: str | None) -> str:
    """保存済みpromptの読み込み。"""
    global current_prompt

    if not prompt_name:
        raise gr.Error("読み込む声を選択してください。")

    current_prompt = engine.load_prompt(prompt_name)
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


def build_ui() -> gr.Blocks:
    """Gradio UIを構築する。"""
    with gr.Blocks(title="Voice Clone TTS", theme=gr.themes.Soft()) as demo:
        gr.Markdown("# Voice Clone TTS\n自分の声をアップロードして、任意のテキストを音読させよう")

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
            gr.Markdown("### Step 2: テキストを入力")
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
            mp3_download = gr.File(label="MP3ダウンロード", visible=False)
            gen_spectrogram = gr.Image(label="生成音声スペクトログラム", visible=True)

        with gr.Group():
            gr.Markdown("### 登録済みの声")
            saved_prompts = gr.Dropdown(
                label="保存済みの声を選択",
                choices=engine.list_prompts() if engine else [],
            )
            load_btn = gr.Button("読み込む")
            delete_btn = gr.Button("削除", variant="stop")
            action_status = gr.Textbox(label="ステータス", interactive=False)

        # イベントバインディング
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
            inputs=[text_input, language_select],
            outputs=[audio_output, mp3_download, gen_spectrogram],
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

    return demo


if __name__ == "__main__":
    engine = VoiceCloneEngine()
    demo = build_ui()
    demo.launch(
        server_name=config["server"]["host"],
        server_port=config["server"]["port"],
    )
