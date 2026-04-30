# GPT-SoVITS セットアップ（Windows / PowerShell）
#
# 使い方:
#   powershell -ExecutionPolicy Bypass -File setup_sovits.ps1 -Device CU128 -Source HF
#   powershell -ExecutionPolicy Bypass -File setup_sovits.ps1 -Device CPU   -Source HF
#
# 必要環境:
#   - uv (https://docs.astral.sh/uv/)
#   - git, ffmpeg

param(
    [Parameter(Mandatory=$true)]
    [ValidateSet("CU126", "CU128", "CPU")]
    [string]$Device,

    [ValidateSet("HF", "HF-Mirror")]
    [string]$Source = "HF",

    [string]$PythonVersion = "3.10"
)

$ErrorActionPreference = "Stop"

function Info($msg)  { Write-Host "[INFO]  $msg" -ForegroundColor Green }
function Warn($msg)  { Write-Host "[WARN]  $msg" -ForegroundColor Yellow }
function Done($msg)  { Write-Host "[DONE]  $msg" -ForegroundColor Cyan }
function Fail($msg)  { Write-Host "[ERROR] $msg" -ForegroundColor Red; exit 1 }

# ===== パス =====
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$SovitsDir = Join-Path $ScriptDir "GPT-SoVITS"
$VenvDir   = Join-Path $SovitsDir ".venv_sovits"

# ===== モデル取得元 =====
switch ($Source) {
    "HF"        { $Base = "https://huggingface.co/XXXXRT/GPT-SoVITS-Pretrained/resolve/main" }
    "HF-Mirror" { $Base = "https://hf-mirror.com/XXXXRT/GPT-SoVITS-Pretrained/resolve/main" }
}
$PretrainedUrl = "$Base/pretrained_models.zip"
$G2pwUrl       = "$Base/G2PWModel.zip"
$NltkUrl       = "$Base/nltk_data.zip"
$OpenJtalkUrl  = "$Base/open_jtalk_dic_utf_8-1.11.tar.gz"

# ===== 必須コマンド =====
function Require-Cmd($cmd) {
    if (-not (Get-Command $cmd -ErrorAction SilentlyContinue)) {
        Fail "$cmd が見つかりません。インストールしてください。"
    }
}
Require-Cmd uv
Require-Cmd git
Require-Cmd ffmpeg
Require-Cmd tar  # Windows 10+ には標準搭載

# ===== Step 1: GPT-SoVITS clone =====
if (-not (Test-Path (Join-Path $SovitsDir "webui.py"))) {
    Info "GPT-SoVITS を clone 中..."
    git clone --depth 1 https://github.com/RVC-Boss/GPT-SoVITS.git $SovitsDir
    Done "clone 完了"
} else {
    Info "GPT-SoVITS は既に存在します。skip"
}

Set-Location $SovitsDir

# ===== Step 2: venv =====
if (-not (Test-Path $VenvDir)) {
    Info "venv を作成中 (Python $PythonVersion)..."
    uv venv --python $PythonVersion $VenvDir
    Done "venv 作成完了"
} else {
    Info "venv は既に存在します。skip"
}

$env:VIRTUAL_ENV = $VenvDir
$env:PATH = "$VenvDir\Scripts;$env:PATH"

# ===== Step 3: PyTorch =====
Info "PyTorch インストール (Device=$Device)..."
# torch/torchaudio は 2.6 系に固定（torchaudio 2.10+ は torchcodec → 動的 libav を要求するため）
$TorchVer = "2.6.0"
switch ($Device) {
    "CU128" { uv pip install "torch==$TorchVer" "torchaudio==$TorchVer" --index-url "https://download.pytorch.org/whl/cu128" }
    "CU126" { uv pip install "torch==$TorchVer" "torchaudio==$TorchVer" --index-url "https://download.pytorch.org/whl/cu126" }
    "CPU"   { uv pip install "torch==$TorchVer" "torchaudio==$TorchVer" --index-url "https://download.pytorch.org/whl/cpu" }
}
Done "PyTorch インストール完了"

# ===== Step 4: 依存パッケージ =====
Info "extra-req.txt を --no-deps でインストール..."
uv pip install -r extra-req.txt --no-deps
Info "requirements.txt をインストール..."
uv pip install -r requirements.txt
Done "依存パッケージ インストール完了"

# ===== Step 5: 事前学習モデル =====
function Download-And-Unzip($url, $targetDir) {
    $zipName = Split-Path $url -Leaf
    Info "Downloading $zipName..."
    Invoke-WebRequest -Uri $url -OutFile $zipName
    Expand-Archive -Path $zipName -DestinationPath $targetDir -Force
    Remove-Item $zipName -Force
}

if (-not (Test-Path "$SovitsDir\GPT_SoVITS\pretrained_models\sv")) {
    Download-And-Unzip $PretrainedUrl "$SovitsDir\GPT_SoVITS"
    Done "pretrained_models 取得完了"
} else {
    Info "pretrained_models は既に存在します。skip"
}

if (-not (Test-Path "$SovitsDir\GPT_SoVITS\text\G2PWModel")) {
    Download-And-Unzip $G2pwUrl "$SovitsDir\GPT_SoVITS\text"
    Done "G2PWModel 取得完了"
} else {
    Info "G2PWModel は既に存在します。skip"
}

# ===== Step 6: NLTK + Open JTalk =====
$PyPrefix         = & "$VenvDir\Scripts\python.exe" -c "import sys; print(sys.prefix)"
$PyopenjtalkPrefix = & "$VenvDir\Scripts\python.exe" -c "import os, pyopenjtalk; print(os.path.dirname(pyopenjtalk.__file__))"

if (-not (Test-Path "$PyPrefix\nltk_data")) {
    Info "NLTK data ダウンロード中..."
    Invoke-WebRequest -Uri $NltkUrl -OutFile nltk_data.zip
    Expand-Archive -Path nltk_data.zip -DestinationPath $PyPrefix -Force
    Remove-Item nltk_data.zip -Force
    Done "NLTK data 取得完了"
} else {
    Info "NLTK data は既に存在します。skip"
}

if (-not (Test-Path "$PyopenjtalkPrefix\open_jtalk_dic_utf_8-1.11")) {
    Info "Open JTalk dict ダウンロード中..."
    Invoke-WebRequest -Uri $OpenJtalkUrl -OutFile open_jtalk_dic.tar.gz
    tar -xzf open_jtalk_dic.tar.gz -C $PyopenjtalkPrefix
    Remove-Item open_jtalk_dic.tar.gz -Force
    Done "Open JTalk dict 取得完了"
} else {
    Info "Open JTalk dict は既に存在します。skip"
}

Write-Host ""
Done "GPT-SoVITS セットアップ完了"
Write-Host "  venv: $VenvDir"
