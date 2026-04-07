"""音声ファイル処理ユーティリティ"""

import logging
import os
import subprocess
import tempfile
import wave

import torch
import numpy as np
import yaml
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import librosa
import librosa.display

logger = logging.getLogger(__name__)

# 設定ファイル読み込み
with open("config.yaml", "r", encoding="utf-8") as f:
    _config = yaml.safe_load(f)

# FFmpegのPATHを自動設定
_ffmpeg_dir = _config["paths"].get("ffmpeg_dir", "")
if _ffmpeg_dir and os.path.isdir(_ffmpeg_dir) and _ffmpeg_dir not in os.environ.get("PATH", ""):
    os.environ["PATH"] = _ffmpeg_dir + os.pathsep + os.environ.get("PATH", "")
    logger.info(f"FFmpegのPATHを追加しました: {_ffmpeg_dir}")

_SAMPLE_RATE = _config["audio"]["sample_rate"]
_MIN_SEC = _config["audio"]["min_duration_sec"]
_MAX_SEC = _config["audio"]["max_duration_sec"]
_WHISPER_MODEL_SIZE = _config["whisper"]["model_size"]


def _resolve_whisper_device() -> tuple[str, str]:
    """Whisperのデバイスとcompute_typeを自動判定する。faster-whisperはMPS非対応のためCUDAかCPU。"""
    cfg_device = _config["whisper"]["device"]
    cfg_compute = _config["whisper"]["compute_type"]

    if cfg_device != "auto" and cfg_compute != "auto":
        return cfg_device, cfg_compute

    if torch.cuda.is_available():
        return "cuda", "float16"
    else:
        return "cpu", "float32"


_WHISPER_DEVICE, _WHISPER_COMPUTE_TYPE = _resolve_whisper_device()


def convert_to_wav(input_path: str, output_path: str, sample_rate: int = _SAMPLE_RATE) -> str:
    """任意の音声ファイルを16kHz WAVに変換。FFmpegを使用。"""
    cmd = [
        "ffmpeg", "-y",
        "-i", input_path,
        "-ar", str(sample_rate),
        "-ac", "1",
        "-sample_fmt", "s16",
        output_path,
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    except FileNotFoundError:
        raise RuntimeError(
            "FFmpegが見つかりません。インストールしてください: https://ffmpeg.org/download.html"
        )
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"FFmpegでの変換に失敗しました: {e.stderr}")
    return output_path


def get_audio_duration(wav_path: str) -> float:
    """WAVファイルの長さ（秒）を返す。"""
    with wave.open(wav_path, "r") as wf:
        frames = wf.getnframes()
        rate = wf.getframerate()
        return frames / rate


def validate_audio(wav_path: str, min_sec: float = _MIN_SEC, max_sec: float = _MAX_SEC) -> tuple[bool, str]:
    """音声ファイルのバリデーション。(is_valid, message)を返す。"""
    duration = get_audio_duration(wav_path)

    if duration < min_sec:
        return False, f"音声が短すぎます（{duration:.1f}秒）。{min_sec}秒以上の音声をアップロードしてください。"
    if duration > max_sec:
        return False, f"音声が長すぎます（{duration:.1f}秒）。{max_sec}秒以下の音声をアップロードしてください。"

    # RMSによる簡易ノイズチェック
    with wave.open(wav_path, "r") as wf:
        raw = wf.readframes(wf.getnframes())
        samples = np.frombuffer(raw, dtype=np.int16).astype(np.float32)
        rms = np.sqrt(np.mean(samples ** 2))

    if rms < 100:
        return True, f"警告: 音声のレベルが非常に低いです（RMS={rms:.0f}）。マイクの音量を確認してください。"

    return True, f"OK（{duration:.1f}秒, RMS={rms:.0f}）"


def transcribe_audio(wav_path: str, language: str = "ja") -> str:
    """faster-whisperで音声を書き起こし。テキストを返す。"""
    from faster_whisper import WhisperModel

    logger.info(f"Whisperモデル（{_WHISPER_MODEL_SIZE}）をロード中...")
    model = WhisperModel(
        _WHISPER_MODEL_SIZE,
        device=_WHISPER_DEVICE,
        compute_type=_WHISPER_COMPUTE_TYPE,
    )

    logger.info("書き起こし実行中...")
    segments, _info = model.transcribe(wav_path, language=language)
    text = "".join(seg.text for seg in segments).strip()

    # メモリ節約のためモデルを解放
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    logger.info(f"書き起こし完了: {text}")

    return text


def generate_spectrogram(audio_path: str, title: str = "スペクトログラム") -> str:
    """音声ファイルからメルスペクトログラム画像を生成し、一時ファイルパスを返す。"""
    y, sr = librosa.load(audio_path, sr=None)
    mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=8000)
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)

    fig, ax = plt.subplots(figsize=(10, 3))
    img = librosa.display.specshow(
        mel_spec_db, sr=sr, x_axis="time", y_axis="mel",
        fmax=8000, ax=ax, cmap="magma",
    )
    fig.colorbar(img, ax=ax, format="%+2.0f dB")
    ax.set_title(title)
    ax.set_xlabel("時間 (秒)")
    ax.set_ylabel("周波数 (Hz)")
    fig.tight_layout()

    tmp_path = tempfile.mktemp(suffix=".png")
    fig.savefig(tmp_path, dpi=120)
    plt.close(fig)
    return tmp_path


def split_audio_by_silence(
    audio_path: str,
    output_dir: str,
    min_segment_sec: float = 3.0,
    max_segment_sec: float = 15.0,
    silence_thresh_db: float = -40.0,
    min_silence_ms: int = 500,
) -> list[dict]:
    """
    長尺音声を無音区間で分割し、各セグメントをWAVで保存する。
    戻り値: [{"path": "segment_001.wav", "start": 0.0, "end": 5.2}, ...]
    """
    from pydub import AudioSegment
    from pydub.silence import split_on_silence

    audio = AudioSegment.from_file(audio_path)
    chunks = split_on_silence(
        audio,
        min_silence_len=min_silence_ms,
        silence_thresh=silence_thresh_db,
        keep_silence=200,
    )

    # 短すぎるチャンクを結合、長すぎるチャンクを強制分割
    merged_chunks = []
    buffer = AudioSegment.empty()
    for chunk in chunks:
        buffer += chunk
        if len(buffer) >= min_segment_sec * 1000:
            # 長すぎる場合は強制分割
            while len(buffer) > max_segment_sec * 1000:
                merged_chunks.append(buffer[:int(max_segment_sec * 1000)])
                buffer = buffer[int(max_segment_sec * 1000):]
            if len(buffer) >= min_segment_sec * 1000:
                merged_chunks.append(buffer)
                buffer = AudioSegment.empty()

    # 残りのバッファ処理
    if len(buffer) > 0:
        if merged_chunks and len(buffer) < min_segment_sec * 1000:
            merged_chunks[-1] += buffer
        else:
            merged_chunks.append(buffer)

    os.makedirs(output_dir, exist_ok=True)
    segments = []
    current_pos = 0.0
    for i, chunk in enumerate(merged_chunks):
        duration = len(chunk) / 1000.0
        seg_path = os.path.join(output_dir, f"segment_{i:04d}.wav")
        chunk.export(seg_path, format="wav", parameters=["-ar", "24000", "-ac", "1"])
        segments.append({
            "path": seg_path,
            "start": current_pos,
            "end": current_pos + duration,
            "duration": duration,
        })
        current_pos += duration

    logger.info(f"音声を{len(segments)}セグメントに分割しました（出力先: {output_dir}）")
    return segments


def transcribe_segments(segments: list[dict], language: str = "ja") -> list[dict]:
    """複数セグメントをまとめてWhisperで書き起こす。各segmentにtextフィールドを追加。"""
    from faster_whisper import WhisperModel

    logger.info(f"Whisperモデル（{_WHISPER_MODEL_SIZE}）をロード中...")
    model = WhisperModel(
        _WHISPER_MODEL_SIZE,
        device=_WHISPER_DEVICE,
        compute_type=_WHISPER_COMPUTE_TYPE,
    )

    for i, seg in enumerate(segments):
        segs, _info = model.transcribe(seg["path"], language=language)
        text = "".join(s.text for s in segs).strip()
        seg["text"] = text
        logger.info(f"セグメント {i + 1}/{len(segments)}: {text[:50]}...")

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return segments


def convert_wav_to_mp3(wav_path: str, mp3_path: str, bitrate: str = "192k") -> str:
    """WAVファイルをMP3に変換する。FFmpegを使用。"""
    cmd = [
        "ffmpeg", "-y",
        "-i", wav_path,
        "-codec:a", "libmp3lame",
        "-b:a", bitrate,
        mp3_path,
    ]
    try:
        subprocess.run(cmd, capture_output=True, text=True, check=True)
    except FileNotFoundError:
        raise RuntimeError("FFmpegが見つかりません。")
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"MP3変換に失敗しました: {e.stderr}")
    return mp3_path
