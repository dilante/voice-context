# Voice Context

[English](README.md) | [中文](README.zh-CN.md)

Windows-first voice prompt editor for AI coding workflows.

Voice Context turns short spoken coding notes into clean Draft prompts that can be pasted directly into Codex, ChatGPT, or another AI coding agent. It supports local, remote, and mixed setups: hold a hotkey, speak, let STT produce text, let a low-latency editor model clean up the prompt, then auto-copy the latest Draft.

> Current public release: stable no-profile edition. Profile/project memory support is planned for a later release.

## Status

- Windows: usable and actively tested.
- macOS/Linux: planned, not fully tested yet.
- Default editor: API provider via OpenAI-compatible chat completions.
- Recommended setup: local Qwen ASR for STT, API model for LLM editing/correction.
- Local Ollama editor is available, but small local models around 3B have not been reliable enough for correction planning in testing.
- Secrets: real API keys stay in `secrets.local.yaml`, which is ignored by git.

## Features

- Hold-to-record listen mode for coding prompts.
- Append, correction, undo, and new-batch Draft workflow.
- Mixed local/remote pipeline: Qwen ASR can run locally while the editor/correction model can use an API.
- API editor presets for OpenAI, DashScope/Qwen, Google Gemini OpenAI-compatible endpoint, DeepSeek, OpenRouter, plus optional local Ollama.
- Qwen ASR provider boundary for local transcription.
- Evaluation cases for append and correction behavior.
- Lightweight context files under `~/.voice_context` by default.
- No raw audio, generated logs, local samples, or secrets are committed.

## Recommended Architecture

For the current release, the recommended balance is:

- STT: local Qwen ASR, because speech recognition benefits from low latency and local privacy.
- LLM editor/correction: API model, because correction planning needs stronger instruction following than the small local models tested so far.
- Optional: local Ollama models for experimentation or offline workflows.

In testing, local models around the 3B class were acceptable for simple append cleanup but not reliable enough for correction planning. That is why the default public config uses an API editor.

## Quick Start on Windows

```powershell
git clone https://github.com/dilante/voice-context.git
cd voice-context

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

Correction and undo can also be triggered by voice, but the trigger rules are intentionally strict:

- Voice undo only triggers when the whole segment is essentially just an undo keyword, such as "undo" or "撤销". If the segment contains other task content, it is treated as normal append text.
- Voice correction requires a correction prefix first, such as "纠错", followed by a short pause, then the correction content. For example: "纠错 ... delete the first sentence" or "纠错 ... 第二句改成更简洁". Without that prefix, a phrase like "change the second line to ..." is treated as append text.
- The keywords are configurable in `config.yaml` under `voice_queue.command_keywords.undo` and `voice_queue.command_keywords.correction_prefixes`.

The keyboard controls are still useful as explicit shortcuts: hold `c` when you want to force correction mode without saying a prefix, and press `u` for deterministic undo.

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

## Configuration Guide

Most behavior is controlled by `config.yaml`:

- `context`: where lightweight session/context JSON files are stored. The default is `~/.voice_context`; set `VOICE_CONTEXT_HOME` to override it.
- `editor`: the active LLM editor preset and runtime options. `provider` selects one entry from `editor_providers`; `timeout_sec`, `temperature`, `correction_max_repair_attempts`, and `no_think` control model calls.
- `secrets`: points to the local secrets file. Keep real API keys in `secrets.local.yaml`; do not put secrets directly in `config.yaml`.
- `editor_providers`: named editor backends. Each preset defines `kind`, `base_url`, `model`, and `api_key_name`. Use this section to switch between OpenAI, DashScope/Qwen, Google, DeepSeek, OpenRouter, or Ollama.
- `prompts`: runtime prompt templates for append normalization and correction planning. Most users do not need to edit these.
- `stt`: selects the speech-to-text provider.
- `stt_providers`: STT backend settings. The recommended current setup is `qwen_local` with `Qwen/Qwen3-ASR-0.6B`.
- `audio`: hotkeys, hold-to-record behavior, and temporary audio location.
- `voice_queue`: language hints and command keywords for voice-triggered undo/correction. Customize `command_keywords.undo` for exact undo phrases and `command_keywords.correction_prefixes` for spoken correction prefixes.
- `output`: clipboard behavior.
- `debug`: replay/eval log settings. Generated logs are ignored by git.

Common edits:

```yaml
editor:
  provider: "openai"  # options: rules, ollama, openai, openrouter, google, deepseek, dashscope

editor_providers:
  openai:
    model: "gpt-4.1-mini"

stt:
  provider: "qwen_local"
```

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

For checks without an API key, `eval-append --editor rules` can verify the basic CLI path.

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
