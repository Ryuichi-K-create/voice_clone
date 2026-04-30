"""GPT-SoVITS 推論サブプロセス（GPT-SoVITS の venv で実行する）

`sovits_engine.py` から subprocess 経由で呼ばれる。
本プロジェクトの venv とは依存関係が衝突するため、独立した venv で実行する設計。

CLI 例:
    python sovits_infer.py \
        --ref_audio /path/to/ref.wav \
        --ref_text  "参照テキスト" \
        --text      "生成したいテキスト" \
        --text_lang ja \
        --output    /path/to/out.wav \
        --device    cuda \
        --version   v2
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

# GPT-SoVITS のルートを sys.path に追加（このスクリプトは voice_clone/scripts/ にある想定）
# 親の親 = voice_clone/、その下に GPT-SoVITS/ がある
_THIS_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _THIS_DIR.parent
_SOVITS_ROOT = _PROJECT_ROOT / "GPT-SoVITS"

if not _SOVITS_ROOT.exists():
    raise FileNotFoundError(f"GPT-SoVITS ディレクトリが見つかりません: {_SOVITS_ROOT}")

sys.path.insert(0, str(_SOVITS_ROOT))
sys.path.insert(0, str(_SOVITS_ROOT / "GPT_SoVITS"))

# 呼び出し時の CWD を保持（後段でユーザー指定の相対パスを解決するため）
_ORIG_CWD = Path.cwd().resolve()

# GPT-SoVITS は CWD 起点で相対パス読み込みする箇所があるので CWD を変更
os.chdir(_SOVITS_ROOT)

from GPT_SoVITS.TTS_infer_pack.TTS import TTS, TTS_Config  # noqa: E402


def _resolve_path(p: str) -> str:
    """ユーザー指定パスを絶対化する（chdir 前の CWD を起点にする）。"""
    path = Path(p)
    if path.is_absolute():
        return str(path)
    return str((_ORIG_CWD / path).resolve())


# 言語コード変換: 本プロジェクト側の表記 → GPT-SoVITS の text_lang
_LANG_MAP = {
    "Japanese": "ja",
    "English":  "en",
    "Chinese":  "zh",
    "Korean":   "ko",
    "ja": "ja", "en": "en", "zh": "zh", "ko": "ko",
}


def _resolve_lang(value: str) -> str:
    if value in _LANG_MAP:
        return _LANG_MAP[value]
    raise ValueError(f"未対応の言語: {value}")


def main() -> None:
    parser = argparse.ArgumentParser(description="GPT-SoVITS 推論")
    parser.add_argument("--ref_audio", required=True, help="参照音声 WAV パス")
    parser.add_argument("--ref_text",  required=True, help="参照音声の書き起こしテキスト")
    parser.add_argument("--ref_lang",  default="Japanese", help="参照テキストの言語")
    parser.add_argument("--text",      required=True, help="生成したいテキスト")
    parser.add_argument("--text_lang", default="Japanese", help="生成テキストの言語")
    parser.add_argument("--output",    required=True, help="出力 WAV パス")
    parser.add_argument("--device",    default="auto", help="cuda / mps / cpu / auto")
    parser.add_argument("--version",   default="v2", choices=["v1", "v2", "v3", "v4", "v2Pro", "v2ProPlus"])
    parser.add_argument("--top_k",       type=int,   default=15)
    parser.add_argument("--top_p",       type=float, default=1.0)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--speed_factor", type=float, default=1.0)
    parser.add_argument("--text_split_method", default="cut5",
                        help="テキスト分割方法（cut0/cut1/cut2/cut3/cut4/cut5）")
    args = parser.parse_args()

    # デバイス自動判定
    import torch
    if args.device == "auto":
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    else:
        device = args.device

    # 半精度は CUDA のみ
    is_half = device == "cuda"

    # TTS 設定
    config_path = _SOVITS_ROOT / "GPT_SoVITS" / "configs" / "tts_infer.yaml"
    tts_config = TTS_Config(str(config_path))
    tts_config.device = device
    tts_config.is_half = is_half
    tts_config.version = args.version

    print(f"[sovits_infer] device={device}, is_half={is_half}, version={args.version}", flush=True)

    # パイプライン構築
    tts_pipeline = TTS(tts_config)

    ref_audio_abs = _resolve_path(args.ref_audio)
    output_abs    = _resolve_path(args.output)

    inputs = {
        "text":              args.text,
        "text_lang":         _resolve_lang(args.text_lang),
        "ref_audio_path":    ref_audio_abs,
        "prompt_text":       args.ref_text,
        "prompt_lang":       _resolve_lang(args.ref_lang),
        "top_k":             args.top_k,
        "top_p":             args.top_p,
        "temperature":       args.temperature,
        "text_split_method": args.text_split_method,
        "batch_size":        1,
        "speed_factor":      args.speed_factor,
        "return_fragment":   False,
        "streaming_mode":    False,
    }

    # run() は generator: (sample_rate, np.ndarray) を yield
    sample_rate = None
    audio_chunks: list[np.ndarray] = []
    for sr, chunk in tts_pipeline.run(inputs):
        sample_rate = sr
        audio_chunks.append(np.asarray(chunk).flatten())

    if sample_rate is None or not audio_chunks:
        raise RuntimeError("音声生成に失敗しました（出力なし）")

    audio = np.concatenate(audio_chunks)

    # int16 → float32 正規化（GPT-SoVITS は int16 出力なので）
    if audio.dtype == np.int16:
        audio = audio.astype(np.float32) / 32768.0

    Path(output_abs).parent.mkdir(parents=True, exist_ok=True)
    sf.write(output_abs, audio, sample_rate)
    print(f"[sovits_infer] saved: {output_abs} ({len(audio)/sample_rate:.2f}s, sr={sample_rate})", flush=True)


if __name__ == "__main__":
    main()
