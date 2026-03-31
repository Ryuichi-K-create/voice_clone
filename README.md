# Voice Clone TTS

自分の声を録音したMP3ファイルをアップロードすると、任意のテキストをその声で音読してくれるシステムです。

## 必要環境

- Python 3.12
- CUDA 12.x + PyTorch 2.x
- FFmpeg
- GPU: VRAM 16GB以上推奨（RTX 4080 Super等）

## セットアップ

```bash
# セットアップスクリプトを実行
bash setup.sh
```

または手動で:

```bash
pip install -r requirements.txt
mkdir -p models docs prompts output
```

## 起動

```bash
python app.py
```

ブラウザで http://localhost:7860 を開いてください。

## 使い方

1. **声をアップロード**: 自分の声を録音したMP3/WAV/M4Aファイル（3~30秒）をアップロード
2. **書き起こし確認**: 自動で書き起こされたテキストを確認・修正
3. **声を登録**: 名前をつけて声を登録
4. **テキスト入力**: 読み上げたいテキストを入力
5. **音声生成**: 「音声を生成」ボタンをクリック

## 技術スタック

- **Qwen3-TTS**: ボイスクローン用TTSモデル
- **faster-whisper**: 参照音声の自動書き起こし
- **FFmpeg**: 音声ファイル変換
- **Gradio**: Web UI
