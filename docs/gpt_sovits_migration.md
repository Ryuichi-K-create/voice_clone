# GPT-SoVITS 導入・解説音声活用・声質調整メモ

現プロジェクトは Qwen3-TTS をベースにしているが、推論速度・配布性・コストの観点から GPT-SoVITS への乗り換えを検討する。本書はその導入手順・解説音声への活用・声質調整機能のメモ。

---

## 1. なぜ GPT-SoVITS か

### Qwen3-TTS との比較

| 観点 | Qwen3-TTS | GPT-SoVITS |
|------|-----------|-----------|
| ライセンス | Apache 2.0 | MIT |
| 商用利用 | ◎ | ◎ |
| ゼロショット | ◎ | ◎（3〜10秒の音声で可） |
| 日本語品質 | 最高 | 高（v3/v4で大幅向上） |
| 推論速度 | 実時間の0.3〜0.5倍（遅い） | 実時間の3〜5倍 |
| 推論VRAM | 8〜16GB | 4〜6GB |
| Fine-tune VRAM | 24GB+ | 8〜12GB |
| Fine-tune 時間 | 数時間〜十数時間 | 30分〜1時間 |
| 10分音声の生成時間 | 20〜40分 | 2〜4分 |

→ **動画解説音声用途では GPT-SoVITS が圧倒的に実用的**。

### 想定運用環境

- 開発機: RTX 4080 Super (16GB) → 余裕で動作
- 本番候補: RTX 5000 Ada Generation (32GB) → オーバースペック気味、Fine-tune も並列実行可能

---

## 2. 導入手順

### 2.1 環境準備

```bash
# 別ディレクトリで GPT-SoVITS を clone
cd ~/Project
git clone https://github.com/RVC-Boss/GPT-SoVITS.git
cd GPT-SoVITS

# Python 環境（Python 3.10 推奨）
uv venv --python 3.10
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# 依存パッケージ
uv pip install -r requirements.txt
```

### 2.2 事前学習モデルのダウンロード

GPT-SoVITS は HuggingFace から事前学習モデルをダウンロードする必要がある。

```bash
# モデル配置先
GPT_SoVITS/pretrained_models/

# 必要なモデル（v4の場合）
- chinese-hubert-base
- chinese-roberta-wwm-ext-large
- s1bert25hz-...
- s2G488k.pth / s2D488k.pth
```

詳細は公式 README の `Get Pretrained Models` を参照。

### 2.3 動作確認

```bash
# WebUI 起動
python webui.py

# ブラウザで http://localhost:9874 を開く
```

---

## 3. 本プロジェクトへの統合方針

### 3.1 既存構造への影響

```
voice_clone/
├── app.py               # Gradio UI（モデル選択UI追加）
├── clone_engine.py      # Qwen3-TTSラッパー（残す）
├── sovits_engine.py     # 新規: GPT-SoVITSラッパー
├── audio_utils.py       # エフェクト機能を拡張
├── config.yaml          # engine 切替設定追加
└── models/              # .pkl + .ckpt（GPT-SoVITS用）
```

### 3.2 config.yaml への追加例

```yaml
engine: gpt_sovits  # qwen3_tts | gpt_sovits

gpt_sovits:
  model_dir: ./models/sovits
  device: auto
  reference_audio: ./samples/my_voice.wav
  reference_text: "こんにちは、これは参照音声です。"
```

### 3.3 共通インターフェース

`clone_engine.py` と `sovits_engine.py` で同じシグネチャの関数を提供：

```python
def load_voice_profile(audio_path: str) -> VoiceProfile: ...
def synthesize(text: str, profile: VoiceProfile) -> np.ndarray: ...
```

`app.py` 側は engine の切替のみ意識する形に。

---

## 4. ゼロショット利用

### 4.1 必要な参照音声

| 項目 | 推奨 |
|------|------|
| 長さ | 3〜10秒 |
| 内容 | クリアな発話、笑い・無音を含まない |
| 音質 | ノイズなし、16kHz 以上、WAV/MP3 |

### 4.2 フロー

```
[mp3アップロード]
   ↓
faster-whisper で書き起こし（既存処理を流用）
   ↓
GPT-SoVITS に reference_audio + reference_text として渡す
   ↓
任意テキスト → 音声生成
```

→ 現プロジェクトと**同じUX**で動作可能。

### 4.3 推論時間の目安（RTX 4080 Super）

| テキスト長 | 生成時間 | 音声長 |
|-----------|---------|--------|
| 短文 | 1〜2秒 | 5〜10秒 |
| 段落 | 5〜10秒 | 30秒〜1分 |
| 10分動画分 | 1〜3分 | 10分 |

---

## 5. Fine-tuning（任意）

### 5.1 ゼロショットとの違い

| | ゼロショット | Fine-tune |
|--|------------|-----------|
| 準備時間 | 即座 | 30分〜1時間 |
| 声の似度 | 80〜90% | 95%以上 |
| 長文安定性 | 中 | 高 |
| 推論速度 | 同じ | 同じ（参照エンコード分のみ短縮） |

### 5.2 必要素材

- 録音音声: 1〜10分（既存の `docs/finetune_script.md` を流用可能）
- 書き起こしテキスト: 自動（faster-whisper）または手動

### 5.3 学習コマンド

GPT-SoVITS の WebUI 上で完結する。

```
1. 1A: データセット前処理（音声分割・ノイズ除去）
2. 1B: 書き起こしテキスト生成
3. 1C: 学習データ整形
4. 2A: SoVITS 学習（音色担当）
5. 2B: GPT 学習（イントネーション担当）
```

各ステップ 5〜15分。RTX 4080 Super で全工程 1 時間以内。

---

## 6. 解説音声（動画ナレーション）への活用

### 6.1 推奨ワークフロー

```
1. 自分の声を5〜10分録音（クリアな環境）
2. GPT-SoVITS にゼロショットまたは Fine-tune で登録
3. 動画台本をテキストファイル化
4. バッチスクリプトで一括生成
5. 動画編集ソフトに取り込み
```

### 6.2 バッチ生成スクリプト例

```python
# scripts/batch_generate.py
from sovits_engine import load_voice_profile, synthesize
import soundfile as sf
from pathlib import Path

profile = load_voice_profile("models/my_voice.pkl")
script_path = Path("scripts/narration.txt")
output_dir = Path("output/narration")
output_dir.mkdir(parents=True, exist_ok=True)

for i, line in enumerate(script_path.read_text(encoding="utf-8").splitlines()):
    if not line.strip():
        continue
    audio = synthesize(line, profile)
    sf.write(output_dir / f"{i:04d}.wav", audio, 32000)
```

### 6.3 品質を上げる小技

| 課題 | 対策 |
|------|------|
| イントネーション崩れ | 句読点を多めに、漢字をひらがな化 |
| 数字・英語の発音ブレ | 台本にカタカナ読みを併記 |
| 長時間音声の声色変化 | 1〜2分単位で生成して結合 |
| 参照音声の品質不足 | ノイズ除去・音量正規化を事前実施 |

---

## 7. 声質調整機能（追加機能）

### 7.1 アプローチ2系統

```
[A] 後処理（生成後にエフェクト加工）
    → 簡単・即実装可能

[B] 生成時制御（モデル側で表現を変える）
    → 高品質・モデル依存
```

### 7.2 後処理エフェクト一覧

| エフェクト | 用途 | ライブラリ |
|-----------|------|----------|
| ピッチシフト | キー上下 | `librosa` / `pyrubberband` |
| タイムストレッチ | 速度変更（音程維持） | `librosa` |
| フォルマントシフト | 性別・年齢感 | `pyworld` |
| リバーブ | 空間感・恐怖感 | `pedalboard` |
| EQ | 低音強調・こもらせ | `pedalboard` |
| ディストーション | ロボット・恐怖 | `pedalboard` |
| コーラス | 不気味・神秘的 | `pedalboard` |

### 7.3 「怖い感じ」のレシピ例

```python
import librosa
from pedalboard import Pedalboard, Reverb, LowpassFilter, Distortion

# 1. ピッチを下げる（-3〜-5半音）
audio = librosa.effects.pitch_shift(audio, sr=sr, n_steps=-4)

# 2. エフェクトチェーン
board = Pedalboard([
    LowpassFilter(cutoff_frequency_hz=3000),  # こもらせる
    Distortion(drive_db=10),                  # 歪ませる
    Reverb(room_size=0.8, damping=0.5),       # 残響
])
scary_audio = board(audio, sr)
```

### 7.4 プリセット案

| プリセット | パラメータ |
|-----------|----------|
| 通常 | エフェクトなし |
| 怖い | ピッチ-4 / ローパス3kHz / 軽いリバーブ |
| かわいい | ピッチ+3 / 速度1.1倍 |
| 老人 | ピッチ-2 / フォルマント-15% |
| ロボット | コーラス + ディストーション |
| 神秘的 | ピッチ-1 / リバーブ強め |

### 7.5 UI 拡張イメージ

```
[テキスト入力]
[参照音声選択]

[ピッチ]   -12 ━━━●━━━ +12 半音
[速度]     0.5 ━━━●━━━ 2.0
[こもり]   弱 ━━━●━━━ 強
[残響]     なし ━━━●━━━ 強
[歪み]     なし ━━━●━━━ 強

[プリセット] 通常 / 怖い / かわいい / 老人 / ロボット / 神秘的

[生成ボタン]
```

### 7.6 生成時制御（参照音声の使い分け）

GPT-SoVITS は参照音声の感情に強く影響される。

```
通常版モデル: 普段の声でクローン
怖い版モデル: 低めの声・抑揚少なめでクローン
楽しい版モデル: 明るく抑揚大きめでクローン
```

各モデルを `.pkl` で保存しておき、UI 上のドロップダウンで切替。

---

## 8. 段階的実装プラン

### Phase 1: 環境構築・動作検証（1〜2日）
- GPT-SoVITS 単体での動作確認
- 自分の声でゼロショット品質を確認
- Qwen3-TTS との品質・速度比較

### Phase 2: プロジェクト統合（2〜3日）
- `sovits_engine.py` 実装
- `config.yaml` で engine 切替
- `app.py` UI 改修

### Phase 3: バッチ生成スクリプト（1日）
- 台本ファイル → 連続音声出力
- 動画編集向けの命名規則

### Phase 4: 声質調整機能（2〜3日）
- `audio_utils.py` に `apply_effects()` 実装
- プリセット定義
- UI スライダー追加

### Phase 5: Fine-tune 対応（任意・1〜2日）
- Fine-tune 結果を `.pkl` 化して既存フローに乗せる
- 通常版・感情別版モデルの管理

---

## 9. ライセンス・配布上の注意

| 項目 | 内容 |
|------|------|
| GPT-SoVITS 本体 | MIT License → 商用利用 ◎ |
| 事前学習モデル | 各モデルのライセンスを個別確認（多くは MIT/Apache） |
| 配布アプリへの組込 | 表示義務のみで OK |
| ボイスクローン倫理 | 利用規約で「自分の声のみ」「悪用禁止」明記 |

---

## 10. 参考リンク

- GPT-SoVITS 公式: https://github.com/RVC-Boss/GPT-SoVITS
- HuggingFace モデル: https://huggingface.co/lj1995/GPT-SoVITS
- pedalboard（Spotify製エフェクト）: https://github.com/spotify/pedalboard
