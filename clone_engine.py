"""Qwen3-TTSラッパー（モデル管理・音声生成）"""

import logging
import os
import pickle
import re

import numpy as np
import torch
import yaml
from qwen_tts import Qwen3TTSModel

import audio_utils

logger = logging.getLogger(__name__)

# 設定ファイル読み込み
with open("config.yaml", "r", encoding="utf-8") as f:
    _config = yaml.safe_load(f)


def _resolve_device(cfg_device: str) -> str:
    """デバイスを自動判定する。CUDA → MPS → CPU の順。"""
    if cfg_device != "auto":
        return cfg_device
    if torch.cuda.is_available():
        return "cuda:0"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _resolve_dtype(cfg_dtype: str, device: str) -> torch.dtype:
    """dtypeを自動判定する。CUDAはbfloat16、MPS/CPUはfloat32。"""
    if cfg_dtype != "auto":
        return getattr(torch, cfg_dtype)
    if device.startswith("cuda"):
        return torch.bfloat16
    return torch.float32


class VoiceCloneEngine:
    def __init__(
        self,
        model_name: str = _config["model"]["name"],
        device: str = _config["model"]["device"],
    ):
        """モデルをロード。初回はHugging Faceから自動ダウンロード。"""
        self.device = _resolve_device(device)
        dtype = _resolve_dtype(_config["model"]["dtype"], self.device)
        logger.info(f"Qwen3-TTSモデル（{model_name}）をロード中... デバイス={self.device}, dtype={dtype}")

        # FlashAttention2の自動判定（CUDA専用）
        attn_impl = "sdpa"
        if self.device.startswith("cuda"):
            try:
                from flash_attn import flash_attn_func  # noqa: F401
                attn_impl = "flash_attention_2"
                logger.info("FlashAttention2を使用します")
            except ImportError:
                logger.info("SDPAにフォールバックします（FlashAttention2未検出）")
        else:
            logger.info(f"SDPAを使用します（デバイス: {self.device}）")

        self.model = Qwen3TTSModel.from_pretrained(
            model_name,
            device_map=self.device,
            dtype=dtype,
            attn_implementation=attn_impl,
        )
        self.prompts_dir = _config["paths"]["prompts_dir"]
        self.max_new_tokens = _config["generation"]["max_new_tokens"]
        self.chunk_max_chars = _config["generation"]["chunk_max_chars"]

        os.makedirs(self.prompts_dir, exist_ok=True)
        logger.info("Qwen3-TTSモデルのロード完了")

    def create_voice_prompt(self, ref_audio: str, ref_text: str | None = None) -> dict:
        """
        参照音声からvoice clone promptを生成。
        ref_textがNoneの場合、faster-whisperで自動書き起こしを行う。
        """
        if ref_text is None or ref_text.strip() == "":
            logger.info("参照テキスト未指定のため、自動書き起こしを実行します")
            ref_text = audio_utils.transcribe_audio(ref_audio)

        logger.info(f"Voice clone promptを生成中（参照テキスト: {ref_text[:30]}...）")
        prompt = self.model.create_voice_clone_prompt(
            ref_audio=ref_audio,
            ref_text=ref_text,
            x_vector_only_mode=False,
        )
        logger.info("Voice clone promptの生成完了")
        return prompt

    def save_prompt(self, prompt: dict, name: str) -> str:
        """promptをpickleで prompts/{name}.pkl に保存。パスを返す。"""
        path = os.path.join(self.prompts_dir, f"{name}.pkl")
        with open(path, "wb") as f:
            pickle.dump(prompt, f)
        logger.info(f"Promptを保存しました: {path}")
        return path

    def load_prompt(self, name: str) -> dict:
        """保存済みpromptを読み込む。"""
        path = os.path.join(self.prompts_dir, f"{name}.pkl")
        if not os.path.exists(path):
            raise FileNotFoundError(f"Promptが見つかりません: {path}")
        with open(path, "rb") as f:
            return pickle.load(f)

    def list_prompts(self) -> list[str]:
        """保存済みprompt一覧を返す（.pklファイル名から拡張子除去）。"""
        if not os.path.exists(self.prompts_dir):
            return []
        return [
            os.path.splitext(f)[0]
            for f in os.listdir(self.prompts_dir)
            if f.endswith(".pkl")
        ]

    def delete_prompt(self, name: str) -> None:
        """保存済みpromptを削除する。"""
        path = os.path.join(self.prompts_dir, f"{name}.pkl")
        if os.path.exists(path):
            os.remove(path)
            logger.info(f"Promptを削除しました: {path}")

    def generate(self, text: str, prompt: dict, language: str = "Japanese") -> tuple:
        """
        テキストからクローン音声を生成。
        (numpy_array, sample_rate) を返す。
        """
        audio, sample_rate = self.model.generate_voice_clone(
            text=text,
            language=language,
            voice_clone_prompt=prompt,
        )
        return audio, sample_rate

    def generate_long(
        self,
        text: str,
        prompt: dict,
        language: str = "Japanese",
        max_chars: int | None = None,
    ) -> tuple:
        """
        長文テキストを分割して生成し、結合して返す。
        句読点（。！？\n）で分割し、各チャンクをgenerate()で生成。
        チャンク間に0.3秒の無音を挿入。
        """
        if max_chars is None:
            max_chars = self.chunk_max_chars

        # 句読点で分割
        chunks = re.split(r"(?<=[。！？\n])", text)
        chunks = [c.strip() for c in chunks if c.strip()]

        # max_charsを超えるチャンクをさらに分割
        final_chunks = []
        for chunk in chunks:
            while len(chunk) > max_chars:
                # 句点がなければmax_chars位置で強制分割
                split_pos = chunk[:max_chars].rfind("、")
                if split_pos == -1:
                    split_pos = max_chars
                else:
                    split_pos += 1  # 「、」を含める
                final_chunks.append(chunk[:split_pos])
                chunk = chunk[split_pos:]
            if chunk:
                final_chunks.append(chunk)

        if not final_chunks:
            final_chunks = [text]

        logger.info(f"テキストを{len(final_chunks)}チャンクに分割して生成します")

        audio_parts = []
        sample_rate = None

        for i, chunk in enumerate(final_chunks):
            logger.info(f"チャンク {i + 1}/{len(final_chunks)}: {chunk[:30]}...")
            audio, sr = self.generate(chunk, prompt, language)
            audio = np.asarray(audio).flatten()
            if sample_rate is None:
                sample_rate = sr
            audio_parts.append(audio)

            # チャンク間に0.3秒の無音を挿入（最後のチャンク以外）
            if i < len(final_chunks) - 1:
                silence = np.zeros(int(sr * 0.3), dtype=audio.dtype)
                audio_parts.append(silence)

        combined = np.concatenate(audio_parts)
        return combined, sample_rate
