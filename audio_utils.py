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
