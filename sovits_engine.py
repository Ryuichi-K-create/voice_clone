"""GPT-SoVITS ラッパー（subprocess 経由）

本プロジェクトの venv とは別に、GPT-SoVITS/.venv_sovits の Python で
scripts/sovits_infer.py を呼び出す。依存関係衝突回避のための分離。

提供 API:
    SovitsEngine.create_voice_prompt(audio_path, ref_text=None) -> dict
    SovitsEngine.save_prompt(prompt, name) -> str
    SovitsEngine.load_prompt(name) -> dict
    SovitsEngine.list_prompts() -> list[str]
    SovitsEngine.delete_prompt(name) -> None
    SovitsEngine.generate(text, prompt, language) -> (np.ndarray, int)
    SovitsEngine.generate_long(text, prompt, language) -> (np.ndarray, int)
"""

from __future__ import annotations

import json
import logging
import os
import platform
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf
import yaml

import audio_utils

logger = logging.getLogger(__name__)

with open("config.yaml", "r", encoding="utf-8") as f:
    _config = yaml.safe_load(f)

_PROJECT_ROOT = Path(__file__).resolve().parent
_SOVITS_ROOT = _PROJECT_ROOT / "GPT-SoVITS"
_INFER_SCRIPT = _PROJECT_ROOT / "scripts" / "sovits_infer.py"


def _resolve_sovits_python() -> Path:
    """GPT-SoVITS 用 venv の python 実行ファイルパスを返す。"""
    cfg = _config.get("gpt_sovits", {})
    custom = cfg.get("python_executable")
    if custom:
        p = Path(custom)
        if p.exists():
            return p

    venv_dir = _SOVITS_ROOT / ".venv_sovits"
    if platform.system() == "Windows":
        return venv_dir / "Scripts" / "python.exe"
    return venv_dir / "bin" / "python"


class SovitsEngine:
    """GPT-SoVITS 推論を subprocess 経由で呼ぶエンジン。"""

    def __init__(self) -> None:
        cfg = _config.get("gpt_sovits", {})
        self.device: str = cfg.get("device", "auto")
        self.version: str = cfg.get("version", "v2")
        self.text_split_method: str = cfg.get("text_split_method", "cut5")
        self.chunk_max_chars: int = _config["generation"].get("chunk_max_chars", 100)

        self.prompts_dir = _config["paths"].get("sovits_prompts_dir", "models/sovits")
        os.makedirs(self.prompts_dir, exist_ok=True)

        self.python_executable = _resolve_sovits_python()
        if not self.python_executable.exists():
            logger.warning(
                "GPT-SoVITS の venv が見つかりません: %s\n"
                "setup_sovits.sh / setup_sovits.ps1 を実行してください。",
                self.python_executable,
            )

        if not _INFER_SCRIPT.exists():
            raise FileNotFoundError(f"推論スクリプトが見つかりません: {_INFER_SCRIPT}")

        logger.info(
            "SovitsEngine 初期化: device=%s, version=%s, python=%s",
            self.device, self.version, self.python_executable,
        )

    # ===== Prompt 管理 =====
    # GPT-SoVITS のゼロショット推論は「参照音声 wav + 参照テキスト」が必要。
    # 本プロジェクトでは {wav_path, ref_text} の dict を「prompt」として扱う。

    def create_voice_prompt(self, ref_audio: str, ref_text: str | None = None) -> dict:
        """参照音声から prompt（dict）を生成する。"""
        if ref_text is None or not ref_text.strip():
            logger.info("参照テキスト未指定 → faster-whisper で書き起こし")
            ref_text = audio_utils.transcribe_audio(ref_audio)

        # 参照音声を prompts ディレクトリにコピーして永続化
        return {
            "ref_audio_src": ref_audio,
            "ref_text": ref_text.strip(),
        }

    def save_prompt(self, prompt: dict, name: str) -> str:
        """
        prompt を {prompts_dir}/{name}/ 配下に保存する。
        - ref.wav: 参照音声をコピー
        - meta.json: 参照テキスト等のメタ情報
        """
        target_dir = Path(self.prompts_dir) / name
        target_dir.mkdir(parents=True, exist_ok=True)

        ref_wav = target_dir / "ref.wav"
        src = prompt.get("ref_audio_src") or prompt.get("ref_audio")
        if src and Path(src).resolve() != ref_wav.resolve():
            shutil.copy2(src, ref_wav)

        meta = {
            "ref_text": prompt["ref_text"],
            "version": self.version,
        }
        with open(target_dir / "meta.json", "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)

        logger.info("prompt を保存しました: %s", target_dir)
        return str(target_dir)

    def load_prompt(self, name: str) -> dict:
        """保存済み prompt を読み込む。"""
        target_dir = Path(self.prompts_dir) / name
        meta_path = target_dir / "meta.json"
        ref_wav = target_dir / "ref.wav"
        if not meta_path.exists() or not ref_wav.exists():
            raise FileNotFoundError(f"prompt が見つかりません: {target_dir}")

        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        return {
            "ref_audio": str(ref_wav),
            "ref_text":  meta["ref_text"],
        }

    def list_prompts(self) -> list[str]:
        """保存済み prompt 一覧。"""
        if not os.path.isdir(self.prompts_dir):
            return []
        return sorted([
            d for d in os.listdir(self.prompts_dir)
            if (Path(self.prompts_dir) / d / "meta.json").exists()
        ])

    def delete_prompt(self, name: str) -> None:
        """保存済み prompt を削除する。"""
        target_dir = Path(self.prompts_dir) / name
        if target_dir.exists():
            shutil.rmtree(target_dir)
            logger.info("prompt を削除しました: %s", target_dir)

    # ===== 音声生成 =====

    def _normalize_prompt(self, prompt: dict) -> tuple[str, str]:
        """prompt dict から (ref_audio_path, ref_text) を取り出す。"""
        ref_audio = prompt.get("ref_audio") or prompt.get("ref_audio_src")
        if not ref_audio:
            raise ValueError("prompt に参照音声パスがありません")
        return str(ref_audio), prompt.get("ref_text", "")

    def generate(self, text: str, prompt: dict, language: str = "Japanese") -> tuple[np.ndarray, int]:
        """単一チャンクで音声生成。(audio, sample_rate) を返す。"""
        ref_audio, ref_text = self._normalize_prompt(prompt)

        # 出力先 tmp ファイル
        out_path = Path(tempfile.mktemp(suffix=".wav"))

        cmd = [
            str(self.python_executable),
            str(_INFER_SCRIPT),
            "--ref_audio", ref_audio,
            "--ref_text",  ref_text,
            "--ref_lang",  language,
            "--text",      text,
            "--text_lang", language,
            "--output",    str(out_path),
            "--device",    self.device,
            "--version",   self.version,
            "--text_split_method", self.text_split_method,
        ]

        logger.info("GPT-SoVITS 推論: %s ...", text[:30])
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0:
            logger.error("sovits_infer stderr:\n%s", proc.stderr)
            logger.error("sovits_infer stdout:\n%s", proc.stdout)
            raise RuntimeError(f"GPT-SoVITS 推論に失敗しました (exit={proc.returncode})")
        if proc.stdout:
            logger.debug("sovits_infer stdout:\n%s", proc.stdout)

        audio, sr = sf.read(str(out_path))
        try:
            out_path.unlink(missing_ok=True)
        except Exception:
            pass

        return np.asarray(audio).flatten(), int(sr)

    def generate_long(
        self,
        text: str,
        prompt: dict,
        language: str = "Japanese",
        max_chars: int | None = None,
    ) -> tuple[np.ndarray, int]:
        """
        長文を句読点で分割して個別生成し、結合して返す。
        GPT-SoVITS 内部にも text_split_method があるが、本プロジェクトの
        clone_engine.py と同じ挙動に揃える形で外側でも分割する。
        チャンク間に 0.3 秒の無音を挿入。
        """
        if max_chars is None:
            max_chars = self.chunk_max_chars

        chunks = re.split(r"(?<=[。！？\n])", text)
        chunks = [c.strip() for c in chunks if c.strip()]

        final_chunks: list[str] = []
        for chunk in chunks:
            while len(chunk) > max_chars:
                split_pos = chunk[:max_chars].rfind("、")
                split_pos = max_chars if split_pos == -1 else split_pos + 1
                final_chunks.append(chunk[:split_pos])
                chunk = chunk[split_pos:]
            if chunk:
                final_chunks.append(chunk)

        if not final_chunks:
            final_chunks = [text]

        logger.info("テキストを %d チャンクに分割して生成", len(final_chunks))

        audio_parts: list[np.ndarray] = []
        sample_rate: int | None = None

        for i, chunk in enumerate(final_chunks):
            logger.info("チャンク %d/%d: %s...", i + 1, len(final_chunks), chunk[:30])
            audio, sr = self.generate(chunk, prompt, language)
            audio = np.asarray(audio).flatten()
            if sample_rate is None:
                sample_rate = sr
            audio_parts.append(audio)
            if i < len(final_chunks) - 1:
                silence = np.zeros(int(sr * 0.3), dtype=audio.dtype)
                audio_parts.append(silence)

        return np.concatenate(audio_parts), int(sample_rate)
