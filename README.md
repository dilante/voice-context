# Voice Context Local

[English](README.md) | [中文](README.zh-CN.md)

Windows-first local voice prompt editor for AI coding workflows.

Voice Context Local turns short spoken coding notes into clean Draft prompts that can be pasted directly into Codex, ChatGPT, or another AI coding agent. It is designed for a fast local workflow: hold a hotkey, speak, let STT produce text, let a low-latency API model clean up the prompt, then auto-copy the latest Draft.

> Current public release: clean no-profile MVP. Profile/project memory support is planned for a later release.

## Status

- Windows: usable and actively tested.
- macOS/Linux: planned, not fully tested yet.
- Default editor: API provider via OpenAI-compatible chat completions.
- STT: local Qwen ASR provider is available; API STT presets are present for experimentation.
- Secrets: real API keys stay in `secrets.local.yaml`, which is ignored by git.

## Features

- Hold-to-record listen mode for coding prompts.
- Append, correction, undo, and new-batch Draft workflow.
- API editor presets for OpenAI, DashScope/Qwen, Google Gemini OpenAI-compatible endpoint, DeepSeek, OpenRouter, and local Ollama.
- Local Qwen ASR provider boundary for offline transcription.
- Evaluation cases for append and correction behavior.
- Local-first context files under `~/.voice_context` by default.
- No raw audio, generated logs, local samples, or secrets are committed.

## Quick Start on Windows

```powershell
git clone https://github.com/dilante/voice-context-local-mvp.git
cd voice-context-local-mvp

python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt

copy secrets.example.yaml secrets.local.yaml
```

Edit `secrets.local.yaml` and add the key for the provider you want to use:

```yaml
api_keys:
  openai: "YOUR_OPENAI_API_KEY"
  dashscope: ""
  google: ""
  deepseek: ""
  openrouter: ""
```

The default provider is configured in `config.yaml`:

```yaml
editor:
  provider: "openai"

editor_providers:
  openai:
    model: "gpt-4.1-mini"
```

Run a smoke test:

```powershell
python -m py_compile src/main.py src/context_store.py src/prompt_editor.py src/editor_providers.py src/qwen_stt_adapter.py src/utils.py src/stt_providers.py src/audio_recorder.py src/voice_queue.py
python src/main.py listen --help
python src/main.py eval-append --cases eval_cases/append --editor rules --debug
```

## Using Listen Mode

Start listen mode with the default API editor:

```powershell
python src/main.py listen --editor openai --timeout 8 --debug
```

Controls:

- Hold `Ctrl+Alt+Space`: record a new append segment.
- Hold `c`: record a correction/delete instruction.
- Press `u`: undo the last Draft change.
- Press `n`: start a new batch.
- Press `q`: quit.

The latest accepted Draft is copied to the clipboard when possible.

## Provider Examples

Switch provider/model from the command line:

```powershell
python src/main.py listen --editor dashscope --model qwen-plus --timeout 8 --debug
python src/main.py listen --editor google --model gemini-2.5-flash --timeout 8 --debug
python src/main.py listen --editor deepseek --model deepseek-chat --timeout 12 --debug
python src/main.py listen --editor openrouter --model google/gemini-2.5-flash-lite --timeout 8 --debug
python src/main.py listen --editor ollama --model qwen3:4b --timeout 20 --debug
```

Provider base URLs, model names, and secret key names are configured in `config.yaml`. Only API keys belong in `secrets.local.yaml`.

## Transcription

Local Qwen ASR is configured in `config.yaml`:

```yaml
stt:
  provider: "qwen_local"

stt_providers:
  qwen_local:
    model_name: "Qwen/Qwen3-ASR-0.6B"
    device: "auto"
    dtype: "auto"
```

Transcribe an audio file:

```powershell
python src/main.py transcribe --audio "C:\path\to\sample.wav" --debug
```

The project does not save raw recordings by default. Temporary audio files live under `~/.voice_context/tmp_audio` and are cleaned up after processing unless configured otherwise.

## Evaluation

Append eval:

```powershell
python src/main.py eval-append --cases eval_cases/append --editor openai --timeout 30 --debug
python src/main.py eval-append --cases eval_cases/append_en --editor openai --timeout 30 --debug
```

Correction eval:

```powershell
python src/main.py eval-corrections --cases eval_cases/correction --editor openai --timeout 60 --debug
python src/main.py eval-corrections --cases eval_cases/correction_en --editor openai --timeout 60 --debug
```

For local checks without an API key, `eval-append --editor rules` can verify the basic CLI path.

## Privacy

Ignored local files include:

- `secrets.local.yaml`
- `.env`
- `.tmp_voice_context/`
- `local_samples/`
- `debug_logs/`
- `eval_runs/`
- `*.wav`, `*.mp3`, `*.m4a`

Do not commit API keys, local recordings, transcripts, model files, or private project context.

## Roadmap

- Public profile/project memory mode.
- Better macOS/Linux hotkey and audio validation.
- Packaged Windows installer.
- More STT API provider validation.
- More general community eval cases.
