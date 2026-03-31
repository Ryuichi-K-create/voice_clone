# Voice Clone TTS - CLAUDE.md

## プロジェクト概要

自分の声を録音したMP3ファイルをアップロードすると、任意のテキストをその声で音読してくれるシステム。

## 技術スタック

- Python 3.12
- Qwen3-TTS (`qwen-tts` パッケージ) - ボイスクローン
- faster-whisper - 参照音声の自動書き起こし
- FFmpeg - 音声ファイル変換
- Gradio - Web UI
- 動作環境: Windows 11 + RTX 4080 Super (VRAM 16GB) または macOS + Apple Silicon (MPS)
- デバイス自動検出: CUDA → MPS → CPU の順で選択（config.yamlで `auto` 指定）

## ディレクトリ構成

```
voice-clone-tts/
├── app.py                  # Gradio Web UI（メインエントリポイント）
├── clone_engine.py         # Qwen3-TTSラッパー（モデル管理・音声生成）
├── audio_utils.py          # 音声ファイル処理ユーティリティ
├── config.yaml             # 設定ファイル
├── requirements.txt        # 依存パッケージ
├── setup.sh                # セットアップスクリプト
├── README.md               # 使い方ドキュメント
├── models/                 # 保存済み voice clone prompt (.pkl)
├── docs/                   # ドキュメント（recording_script.md等）
├── prompts/                # プロンプト設計メモ
└── output/                 # 生成された音声ファイル
```

## コーディングルール

- コメントは日本語で書く
- コミットメッセージは日本語で書く
- 型ヒントを使用する
- f-stringを使用する
- printではなくlogging（もしくはGradioのステータス表示）を使う
- 作業の前に確認を取る。自動で作業を開始しない
- 1つのタスクが完了したら要約せず、次の指示を待つ

## 実装上の注意点

- モデルロードは起動時に1回だけ行い、グローバルで保持する
- 長文は句読点（。！？\n）で分割し、各チャンクを個別生成してnumpy.concatenateで結合。チャンク間に0.3秒の無音を挿入
- FlashAttention 2が未インストールでもsdpaにフォールバックする
- VRAM節約のため、Whisperは書き起こし完了後にメモリから解放する
- 生成音声は `output/{timestamp}_{最初の10文字}.wav` で自動保存

## 開発コマンド

```bash
# セットアップ
bash setup.sh
# サーバー起動
uv run python app.py
# ブラウザで http://localhost:7860 を開く
```
