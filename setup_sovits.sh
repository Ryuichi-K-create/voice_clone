#!/usr/bin/env bash
# GPT-SoVITS セットアップ（Linux / macOS）
#
# 使い方:
#   bash setup_sovits.sh --device CU128 --source HF
#   bash setup_sovits.sh --device CPU   --source HF
#   bash setup_sovits.sh --device MPS   --source HF      # macOS Apple Silicon
#
# 必要環境:
#   - uv (https://docs.astral.sh/uv/)
#   - git, wget, unzip, tar, ffmpeg

set -euo pipefail

# ===== 色設定 =====
RESET="\033[0m"
BOLD="\033[1m"
ERROR="\033[1;31m[ERROR]:$RESET "
WARN="\033[1;33m[WARN]:$RESET "
INFO="\033[1;32m[INFO]:$RESET "
DONE="\033[1;34m[DONE]:$RESET "

# ===== パス =====
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SOVITS_DIR="$SCRIPT_DIR/GPT-SoVITS"
VENV_DIR="$SOVITS_DIR/.venv_sovits"

# ===== オプション =====
DEVICE=""
SOURCE=""
PYTHON_VERSION="3.10"

print_help() {
    cat <<EOF
Usage: bash setup_sovits.sh [OPTIONS]

Options:
  --device   CU126|CU128|CPU|MPS    インストール先デバイス (必須)
  --source   HF|HF-Mirror           モデル取得元 (デフォルト: HF)
  --python   3.10|3.11              Python バージョン (デフォルト: 3.10)
  -h, --help                        このヘルプを表示

Examples:
  bash setup_sovits.sh --device CU128 --source HF
  bash setup_sovits.sh --device MPS   --source HF      # macOS
  bash setup_sovits.sh --device CPU   --source HF
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --device) DEVICE="$2"; shift 2 ;;
        --source) SOURCE="$2"; shift 2 ;;
        --python) PYTHON_VERSION="$2"; shift 2 ;;
        -h|--help) print_help; exit 0 ;;
        *) echo -e "${ERROR}Unknown argument: $1"; print_help; exit 1 ;;
    esac
done

if [[ -z "$DEVICE" ]]; then
    echo -e "${ERROR}--device は必須です"
    print_help
    exit 1
fi
SOURCE="${SOURCE:-HF}"

# ===== モデル取得元 URL =====
case "$SOURCE" in
    HF)
        BASE="https://huggingface.co/XXXXRT/GPT-SoVITS-Pretrained/resolve/main"
        ;;
    HF-Mirror)
        BASE="https://hf-mirror.com/XXXXRT/GPT-SoVITS-Pretrained/resolve/main"
        ;;
    *)
        echo -e "${ERROR}--source は HF か HF-Mirror を指定してください"
        exit 1
        ;;
esac
PRETRAINED_URL="$BASE/pretrained_models.zip"
G2PW_URL="$BASE/G2PWModel.zip"
NLTK_URL="$BASE/nltk_data.zip"
OPENJTALK_URL="$BASE/open_jtalk_dic_utf_8-1.11.tar.gz"

# ===== 必須コマンド確認 =====
require_cmd() {
    if ! command -v "$1" &>/dev/null; then
        echo -e "${ERROR}$1 が見つかりません。インストールしてください。"
        exit 1
    fi
}
require_cmd uv
require_cmd git
require_cmd wget
require_cmd unzip
require_cmd tar
require_cmd ffmpeg

# ===== Step 1: GPT-SoVITS リポジトリの clone =====
if [[ ! -d "$SOVITS_DIR/.git" && ! -f "$SOVITS_DIR/webui.py" ]]; then
    echo -e "${INFO}GPT-SoVITS を clone 中..."
    git clone --depth 1 https://github.com/RVC-Boss/GPT-SoVITS.git "$SOVITS_DIR"
    echo -e "${DONE}clone 完了"
else
    echo -e "${INFO}GPT-SoVITS は既に存在します。skip"
fi

cd "$SOVITS_DIR"

# ===== Step 2: venv 作成 =====
if [[ ! -d "$VENV_DIR" ]]; then
    echo -e "${INFO}venv を作成中 (Python $PYTHON_VERSION)..."
    uv venv --python "$PYTHON_VERSION" "$VENV_DIR"
    echo -e "${DONE}venv 作成完了"
else
    echo -e "${INFO}venv は既に存在します。skip"
fi

# uv pip 用の環境変数
export VIRTUAL_ENV="$VENV_DIR"
export PATH="$VENV_DIR/bin:$PATH"

# ===== Step 3: PyTorch インストール =====
echo -e "${INFO}PyTorch インストール (DEVICE=$DEVICE)..."
# torch/torchaudio は 2.6 系に固定（torchaudio 2.10+ は torchcodec → libav 動的リンクを要求し、
# static-build の ffmpeg しか無い環境で詰まるため）。
TORCH_VER="2.6.0"
case "$DEVICE" in
    CU128) uv pip install "torch==$TORCH_VER" "torchaudio==$TORCH_VER" --index-url "https://download.pytorch.org/whl/cu128" ;;
    CU126) uv pip install "torch==$TORCH_VER" "torchaudio==$TORCH_VER" --index-url "https://download.pytorch.org/whl/cu126" ;;
    CPU)   uv pip install "torch==$TORCH_VER" "torchaudio==$TORCH_VER" --index-url "https://download.pytorch.org/whl/cpu" ;;
    MPS)   uv pip install "torch==$TORCH_VER" "torchaudio==$TORCH_VER" ;;  # macOS は PyPI で MPS 対応
    *)
        echo -e "${ERROR}--device は CU126/CU128/CPU/MPS のいずれかを指定してください"
        exit 1
        ;;
esac
echo -e "${DONE}PyTorch インストール完了"

# ===== Step 4: 依存パッケージ =====
echo -e "${INFO}extra-req.txt を --no-deps でインストール..."
uv pip install -r extra-req.txt --no-deps

echo -e "${INFO}requirements.txt をインストール..."
uv pip install -r requirements.txt
echo -e "${DONE}依存パッケージ インストール完了"

# ===== Step 5: 事前学習モデル =====
download_and_unzip() {
    local url="$1"
    local target_dir="$2"
    local zip_name
    zip_name="$(basename "$url")"

    echo -e "${INFO}Downloading $zip_name..."
    wget --tries=10 --read-timeout=60 -c -O "$zip_name" "$url"
    unzip -q -o "$zip_name" -d "$target_dir"
    rm -f "$zip_name"
}

if [[ ! -d "$SOVITS_DIR/GPT_SoVITS/pretrained_models/sv" ]]; then
    download_and_unzip "$PRETRAINED_URL" "$SOVITS_DIR/GPT_SoVITS"
    echo -e "${DONE}pretrained_models 取得完了"
else
    echo -e "${INFO}pretrained_models は既に存在します。skip"
fi

if [[ ! -d "$SOVITS_DIR/GPT_SoVITS/text/G2PWModel" ]]; then
    download_and_unzip "$G2PW_URL" "$SOVITS_DIR/GPT_SoVITS/text"
    echo -e "${DONE}G2PWModel 取得完了"
else
    echo -e "${INFO}G2PWModel は既に存在します。skip"
fi

# ===== Step 6: NLTK + Open JTalk Dict =====
PY_PREFIX="$("$VENV_DIR/bin/python" -c 'import sys; print(sys.prefix)')"
PYOPENJTALK_PREFIX="$("$VENV_DIR/bin/python" -c 'import os, pyopenjtalk; print(os.path.dirname(pyopenjtalk.__file__))')"

if [[ ! -d "$PY_PREFIX/nltk_data" ]]; then
    echo -e "${INFO}NLTK data ダウンロード中..."
    wget --tries=10 --read-timeout=60 -c -O nltk_data.zip "$NLTK_URL"
    unzip -q -o nltk_data.zip -d "$PY_PREFIX"
    rm -f nltk_data.zip
    echo -e "${DONE}NLTK data 取得完了"
else
    echo -e "${INFO}NLTK data は既に存在します。skip"
fi

if [[ ! -d "$PYOPENJTALK_PREFIX/open_jtalk_dic_utf_8-1.11" ]]; then
    echo -e "${INFO}Open JTalk dict ダウンロード中..."
    wget --tries=10 --read-timeout=60 -c -O open_jtalk_dic.tar.gz "$OPENJTALK_URL"
    tar -xzf open_jtalk_dic.tar.gz -C "$PYOPENJTALK_PREFIX"
    rm -f open_jtalk_dic.tar.gz
    echo -e "${DONE}Open JTalk dict 取得完了"
else
    echo -e "${INFO}Open JTalk dict は既に存在します。skip"
fi

echo
echo -e "${DONE}${BOLD}GPT-SoVITS セットアップ完了${RESET}"
echo "  venv: $VENV_DIR"
echo "  動作確認: bash $SCRIPT_DIR/setup_sovits.sh --device $DEVICE --source $SOURCE  （何度実行してもOK）"
