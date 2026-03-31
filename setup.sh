#!/bin/bash
# Voice Clone TTS セットアップスクリプト

set -e

echo "=== Voice Clone TTS セットアップ ==="

# uv確認
if ! command -v uv &> /dev/null; then
    echo "エラー: uvがインストールされていません。"
    echo "インストール: curl -LsSf https://astral.sh/uv/install.sh | sh"
    exit 1
fi

echo "uv: $(uv --version)"

# venv作成
if [ ! -d ".venv" ]; then
    echo ""
    echo "=== Python 3.12 仮想環境を作成中 ==="
    uv venv --python 3.12
fi

# FFmpegインストール
if command -v ffmpeg &> /dev/null; then
    ffmpeg_version=$(ffmpeg -version 2>&1 | head -n 1)
    echo "FFmpeg: $ffmpeg_version"
else
    echo ""
    echo "=== FFmpegをインストール中 ==="
    if [[ "$(uname)" == "Darwin" ]]; then
        brew install ffmpeg
    elif command -v apt &> /dev/null; then
        sudo apt install -y ffmpeg
    elif command -v winget &> /dev/null; then
        winget install FFmpeg
    else
        echo "エラー: FFmpegを自動インストールできません。手動でインストールしてください。"
        exit 1
    fi
fi

# SoXインストール（qwen-ttsが内部で使用）
if command -v sox &> /dev/null; then
    echo "SoX: $(sox --version 2>&1 | head -n 1)"
else
    echo ""
    echo "=== SoXをインストール中 ==="
    if [[ "$(uname)" == "Darwin" ]]; then
        brew install sox
    elif command -v apt &> /dev/null; then
        sudo apt install -y sox
    else
        echo "警告: SoXを自動インストールできません。手動でインストールしてください。"
    fi
fi

# 依存パッケージのインストール
echo ""
echo "=== 依存パッケージをインストール中 ==="
uv pip install -r requirements.txt

# CUDA対応PyTorchのインストール（GPU環境の場合）
echo ""
echo "=== PyTorch CUDA版を確認中 ==="
if python -c "import torch; assert '+cu' in torch.__version__" 2>/dev/null; then
    echo "PyTorch CUDA版: 既にインストール済み"
else
    if [[ "$(uname -s)" == MINGW* ]] || [[ "$(uname -s)" == MSYS* ]] || [[ "$(uname -s)" == CYGWIN* ]] || command -v nvidia-smi &> /dev/null; then
        echo "NVIDIA GPUを検出。PyTorch CUDA版をインストール中..."
        uv pip install --reinstall torch torchaudio --index-url https://download.pytorch.org/whl/cu126
    else
        echo "NVIDIA GPUが検出されませんでした。CPU版のまま使用します。"
    fi
fi

# ディレクトリ作成
mkdir -p models docs prompts output

echo ""
echo "=== セットアップ完了 ==="
echo "起動コマンド: uv run python app.py"
echo "ブラウザで http://localhost:7860 を開いてください"
