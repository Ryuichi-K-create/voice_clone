# Voice Clone TTS - Claude Code 実装指示書

## プロジェクト概要

自分の声を録音したMP3ファイルをアップロードすると、任意のテキストをその声で音読してくれるシステムを構築する。

### ゴール

1. ユーザーが自分の声のMP3ファイル（3〜30秒）を入力する
2. 任意の日本語テキストを入力する
3. ユーザーの声をクローンした音声（WAVファイル）が出力される

### 技術スタック

- Python 3.12
- Qwen3-TTS（`qwen-tts` パッケージ）- ボイスクローン用
- faster-whisper - 参照音声の自動書き起こし用
- FFmpeg - MP3→WAV変換用
- Gradio - Web UI用

### 動作環境

- Windows 11 + RTX 4080 Super（VRAM 16GB）
- CUDA 12.x + PyTorch 2.x

---

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
├── prompts/                # 保存済み voice clone prompt
│   └── .gitkeep
└── output/                 # 生成された音声ファイル
    └── .gitkeep
```

---

## 実装仕様

### 1. `audio_utils.py` - 音声ファイル処理

#### 機能

- MP3/M4A/WAV等の入力ファイルをQwen3-TTS用の16kHz WAVに変換
- FFmpegを使用（`subprocess.run`で呼び出し）
- 音声の長さを検証（3秒〜30秒の範囲でバリデーション）
- 音声のノイズレベルを簡易チェック（RMSが閾値以下なら警告）

#### 関数

```python
def convert_to_wav(input_path: str, output_path: str, sample_rate: int = 16000) -> str:
    """任意の音声ファイルを16kHz WAVに変換。FFmpegを使用。"""

def get_audio_duration(wav_path: str) -> float:
    """WAVファイルの長さ（秒）を返す。"""

def validate_audio(wav_path: str, min_sec: float = 3.0, max_sec: float = 30.0) -> tuple[bool, str]:
    """音声ファイルのバリデーション。(is_valid, message)を返す。"""

def transcribe_audio(wav_path: str, language: str = "ja") -> str:
    """faster-whisperで音声を書き起こし。テキストを返す。"""
```

#### 書き起こしの実装詳細

```python
from faster_whisper import WhisperModel

# モデルは"large-v3"を使用。初回のみダウンロード。
# device="cuda", compute_type="float16" で高速化
# 結果のsegmentsからテキストを結合して返す
```

### 2. `clone_engine.py` - Qwen3-TTSラッパー

#### 機能

- Qwen3-TTS 1.7B Baseモデルのロード・管理
- voice clone promptの生成・保存・読み込み
- テキストからクローン音声を生成

#### クラス設計

```python
class VoiceCloneEngine:
    def __init__(self, model_name: str = "Qwen/Qwen3-TTS-12Hz-1.7B-Base", device: str = "cuda:0"):
        """モデルをロード。初回はHugging Faceから自動ダウンロード。"""
        # self.model = Qwen3TTSModel.from_pretrained(...)
        # dtype=torch.bfloat16
        # attn_implementation: flash_attention_2が使えれば使う、なければsdpa
        # FlashAttention2の利用可否を自動判定すること

    def create_voice_prompt(self, ref_audio: str, ref_text: str | None = None) -> dict:
        """
        参照音声からvoice clone promptを生成。
        ref_textがNoneの場合、faster-whisperで自動書き起こしを行う。
        生成したpromptを返す。
        """
        # ref_textが未指定 → audio_utils.transcribe_audio()で自動取得
        # model.create_voice_clone_prompt()を呼び出し
        # x_vector_only_mode=False（テキストありの方が品質高い）

    def save_prompt(self, prompt: dict, name: str) -> str:
        """promptをpickleで prompts/{name}.pkl に保存。パスを返す。"""

    def load_prompt(self, name: str) -> dict:
        """保存済みpromptを読み込む。"""

    def list_prompts(self) -> list[str]:
        """保存済みprompt一覧を返す（.pklファイル名から拡張子除去）。"""

    def generate(self, text: str, prompt: dict, language: str = "Japanese") -> tuple:
        """
        テキストからクローン音声を生成。
        (numpy_array, sample_rate) を返す。
        """
        # model.generate_voice_clone(
        #     text=text,
        #     language=language,
        #     voice_clone_prompt=prompt,
        # )

    def generate_long(self, text: str, prompt: dict, language: str = "Japanese", max_chars: int = 100) -> tuple:
        """
        長文テキストを分割して生成し、結合して返す。
        句読点（。！？）で分割し、各チャンクをgenerate()で生成。
        numpy.concatenateで結合。
        """
```

#### モデルロード時の注意点

```python
import torch
from qwen_tts import Qwen3TTSModel

# FlashAttention2の自動判定
try:
    from flash_attn import flash_attn_func
    attn_impl = "flash_attention_2"
except ImportError:
    attn_impl = "sdpa"  # PyTorch標準のScaled Dot Product Attention

model = Qwen3TTSModel.from_pretrained(
    model_name,
    device_map=device,
    dtype=torch.bfloat16,
    attn_implementation=attn_impl,
)
```

### 3. `app.py` - Gradio Web UI

#### UI構成

Gradioの`Blocks`を使用。タブは不要、1画面で完結させる。

```
┌─────────────────────────────────────────┐
│  🎙 Voice Clone TTS                     │
├─────────────────────────────────────────┤
│                                         │
│  ■ Step 1: 声をアップロード              │
│  [音声ファイル選択] (MP3/WAV/M4A)        │
│  [書き起こし結果] (自動表示、編集可能)    │
│  [声を登録] ボタン                       │
│  → 登録完了メッセージ                    │
│                                         │
│  ■ Step 2: テキストを入力               │
│  [テキスト入力欄] (複数行)               │
│  [言語選択] Japanese / English / ...     │
│  [生成] ボタン                           │
│  → [音声プレイヤー] 生成結果             │
│  → [ダウンロード] ボタン                 │
│                                         │
│  ■ 登録済みの声                         │
│  [ドロップダウン] 保存済みprompt選択      │
│  [削除] ボタン                           │
│                                         │
└─────────────────────────────────────────┘
```

#### Gradio実装の方針

```python
import gradio as gr

# グローバルにエンジンを保持（起動時にモデルロード）
engine: VoiceCloneEngine = None
current_prompt: dict = None

def on_upload(audio_file):
    """
    1. audio_utils.convert_to_wav()で変換
    2. audio_utils.validate_audio()でバリデーション
    3. audio_utils.transcribe_audio()で書き起こし
    4. 書き起こし結果をテキストボックスに表示
    """

def on_register(audio_file, transcript, prompt_name):
    """
    1. engine.create_voice_prompt()でprompt生成
    2. engine.save_prompt()で保存
    3. current_promptを更新
    4. 成功メッセージを返す
    """

def on_generate(text, language):
    """
    1. current_promptを使ってengine.generate_long()で音声生成
    2. soundfile.write()でWAV保存
    3. Gradioのaudioコンポーネントに返す
    """

def build_ui():
    with gr.Blocks(title="Voice Clone TTS", theme=gr.themes.Soft()) as demo:
        gr.Markdown("# 🎙 Voice Clone TTS\n自分の声をアップロードして、任意のテキストを音読させよう")

        with gr.Group():
            gr.Markdown("### Step 1: 声をアップロード")
            audio_input = gr.Audio(sources=["upload"], type="filepath", label="音声ファイル (MP3/WAV/M4A, 3〜30秒)")
            transcript_box = gr.Textbox(label="書き起こし結果（自動入力・編集可）", lines=2)
            prompt_name_input = gr.Textbox(label="声の名前", value="my_voice", max_lines=1)
            register_btn = gr.Button("声を登録", variant="primary")
            register_status = gr.Textbox(label="ステータス", interactive=False)

        with gr.Group():
            gr.Markdown("### Step 2: テキストを入力")
            text_input = gr.Textbox(label="読み上げたいテキスト", lines=5, placeholder="ここにテキストを入力...")
            language_select = gr.Dropdown(
                choices=["Japanese", "English", "Chinese", "Korean"],
                value="Japanese",
                label="言語"
            )
            generate_btn = gr.Button("音声を生成", variant="primary")
            audio_output = gr.Audio(label="生成結果", type="filepath")

        with gr.Group():
            gr.Markdown("### 登録済みの声")
            saved_prompts = gr.Dropdown(label="保存済みの声を選択", choices=[])
            load_btn = gr.Button("読み込む")
            delete_btn = gr.Button("削除", variant="stop")

        # イベントバインディング
        audio_input.change(fn=on_upload, inputs=[audio_input], outputs=[transcript_box])
        register_btn.click(fn=on_register, inputs=[audio_input, transcript_box, prompt_name_input], outputs=[register_status, saved_prompts])
        generate_btn.click(fn=on_generate, inputs=[text_input, language_select], outputs=[audio_output])
        # load_btn, delete_btn も同様にバインド

    return demo
```

#### 起動

```python
if __name__ == "__main__":
    engine = VoiceCloneEngine()
    demo = build_ui()
    demo.launch(server_name="0.0.0.0", server_port=7860)
```

### 4. `config.yaml`

```yaml
model:
  name: "Qwen/Qwen3-TTS-12Hz-1.7B-Base"
  device: "cuda:0"
  dtype: "bfloat16"

whisper:
  model_size: "large-v3"
  device: "cuda"
  compute_type: "float16"

audio:
  sample_rate: 16000
  min_duration_sec: 3.0
  max_duration_sec: 30.0

generation:
  max_new_tokens: 2048
  chunk_max_chars: 100   # 長文分割時の1チャンクの最大文字数

paths:
  prompts_dir: "prompts"
  output_dir: "output"

server:
  host: "0.0.0.0"
  port: 7860
```

### 5. `requirements.txt`

```
qwen-tts>=0.0.5
torch>=2.0.0
soundfile
pyyaml
gradio>=4.0.0
faster-whisper>=1.0.0
numpy
```

---

## 実装上の注意点

### モデルロードは起動時に1回だけ

Qwen3-TTSのモデルロードには30秒〜1分かかる。`app.py`の起動時に1回だけロードし、グローバル変数またはクラスインスタンスで保持すること。リクエストごとにロードしてはいけない。

### 長文の分割処理

Qwen3-TTSは1回の生成で長すぎるテキストを渡すと品質が劣化する。句読点（`。`、`！`、`？`、`\n`）で分割し、各チャンクを個別に生成してnumpy.concatenateで結合すること。チャンク間に0.3秒程度の無音（numpy.zeros）を挿入すると自然に聞こえる。

### FlashAttention 2

FlashAttention 2がインストールされていない環境でもフォールバック（sdpa）で動作するようにすること。Windowsではflash-attnのインストールが難しい場合がある。`--no-flash-attn`相当のフォールバックを必ず実装すること。

### Whisperのモデルサイズ

VRAM 16GBにQwen3-TTS 1.7B（約6GB）とWhisper large-v3（約3GB）を同時に載せるとVRAMが厳しくなる可能性がある。Whisperは書き起こし完了後にメモリから解放するか、`medium`サイズを使って節約すること。もしくは、Whisperの処理をQwen3-TTSのロード前に行い、完了後にWhisperを解放してからQwen3-TTSをロードする設計にする。

### 生成音声の保存

生成した音声は`output/{timestamp}_{最初の10文字}.wav`の形式で自動保存すること。Gradioのaudioコンポーネントにはファイルパスを渡す。

### エラーハンドリング

- 音声ファイルが短すぎる/長すぎる場合: バリデーションメッセージを表示
- FFmpegが見つからない場合: インストール手順を表示
- VOICEVOXとの共存: ポートの競合はないが、VRAM共有に注意
- モデルダウンロード失敗: リトライとオフラインモードの案内
- 生成中の無限ループ: max_new_tokens=2048でタイムアウト

---

## テスト手順

### 1. 基本動作テスト

```bash
# サーバー起動
python app.py

# ブラウザで http://localhost:7860 を開く
# 1. 自分の声のMP3をアップロード
# 2. 書き起こしが自動表示されることを確認
# 3. 「声を登録」をクリック
# 4. テキスト欄に「バナナは放射線を出しているって知ってましたか？」と入力
# 5. 「音声を生成」をクリック
# 6. 自分の声で読み上げられた音声が再生されることを確認
```

### 2. 品質確認ポイント

- 自分の声にどれくらい似ているか
- 日本語のアクセント・イントネーションは自然か
- 長文（200文字以上）を入力した場合、チャンク間の繋がりは自然か
- 同じテキストを2回生成した時、声質は一貫しているか

---

## コーディングルール

- コメントは日本語で書くこと
- コミットメッセージは日本語で書くこと
- 型ヒントを使用すること
- f-stringを使用すること
- printではなくlogging（もしくはGradioのステータス表示）を使うこと
- 作業の前に確認を取ること。自動で作業を開始しないこと
- 1つのタスクが完了したら要約せず、次の指示を待つこと