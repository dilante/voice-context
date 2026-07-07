# Voice Context Local

[English](README.md) | [中文](README.zh-CN.md)

Voice Context Local 是一个 Windows-first 的本地语音 prompt 编辑工具，用于把短语音编程指令整理成可以直接发给 Codex、ChatGPT 或其他 AI coding agent 的 Draft prompt。

当前公开版本是稳定的无 profile MVP：默认使用 API editor，本地保存轻量 context，支持录音追加、纠错、撤销和自动复制。profile / 项目长期记忆会在后续版本支持。

## 当前状态

- Windows：可用，主要测试平台。
- macOS/Linux：计划支持，尚未完整测试。
- 默认 editor：API provider，OpenAI-compatible chat completions。
- STT：已提供本地 Qwen ASR provider 边界；API STT preset 处于实验阶段。
- 密钥：只放在 `secrets.local.yaml`，不会提交。

## 功能

- 按住热键录音，生成可直接发给 AI coding agent 的 Draft。
- 支持 append、correction、undo 和 new batch 工作流。
- API editor preset 支持 OpenAI、DashScope/Qwen、Google Gemini OpenAI-compatible endpoint、DeepSeek、OpenRouter 和本地 Ollama。
- 提供本地 Qwen ASR provider 边界，用于离线转写。
- 包含 append / correction eval cases。
- 默认本地 context 存在 `~/.voice_context`。
- 不提交原始音频、生成日志、本地 sample 或密钥。

## Windows 部署

```powershell
git clone https://github.com/dilante/voice-context-local-mvp.git
cd voice-context-local-mvp

python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt

copy secrets.example.yaml secrets.local.yaml
```

然后在 `secrets.local.yaml` 填入 API key：

```yaml
api_keys:
  openai: "YOUR_OPENAI_API_KEY"
  dashscope: ""
  google: ""
  deepseek: ""
  openrouter: ""
```

默认 provider 在 `config.yaml` 里改：

```yaml
editor:
  provider: "openai"

editor_providers:
  openai:
    model: "gpt-4.1-mini"
```

运行 smoke test：

```powershell
python -m py_compile src/main.py src/context_store.py src/prompt_editor.py src/editor_providers.py src/qwen_stt_adapter.py src/utils.py src/stt_providers.py src/audio_recorder.py src/voice_queue.py
python src/main.py listen --help
python src/main.py eval-append --cases eval_cases/append --editor rules --debug
```

## 使用 Listen Mode

启动常驻 listen：

```powershell
python src/main.py listen --editor openai --timeout 8 --debug
```

快捷键：

- 按住 `Ctrl+Alt+Space`：录一段新的 append 语音。
- 按住 `c`：录一段纠错或删除指令。
- 按 `u`：撤销上一条 Draft 修改。
- 按 `n`：新建 batch。
- 按 `q`：退出。

成功生成的最新 Draft 会尽量自动复制到剪贴板。

## 切换模型

```powershell
python src/main.py listen --editor dashscope --model qwen-plus --timeout 8 --debug
python src/main.py listen --editor google --model gemini-2.5-flash --timeout 8 --debug
python src/main.py listen --editor deepseek --model deepseek-chat --timeout 12 --debug
python src/main.py listen --editor openrouter --model google/gemini-2.5-flash-lite --timeout 8 --debug
python src/main.py listen --editor ollama --model qwen3:4b --timeout 20 --debug
```

provider 的 base URL、model、api key 名称都在 `config.yaml` 配置；真实 key 只放 `secrets.local.yaml`。

## 转写

本地 Qwen ASR 在 `config.yaml` 中配置：

```yaml
stt:
  provider: "qwen_local"

stt_providers:
  qwen_local:
    model_name: "Qwen/Qwen3-ASR-0.6B"
    device: "auto"
    dtype: "auto"
```

转写本地音频文件：

```powershell
python src/main.py transcribe --audio "C:\path\to\sample.wav" --debug
```

项目默认不保存原始录音。临时音频放在 `~/.voice_context/tmp_audio`，处理后会清理，除非配置为保留。

## 评估

Append eval：

```powershell
python src/main.py eval-append --cases eval_cases/append --editor openai --timeout 30 --debug
python src/main.py eval-append --cases eval_cases/append_en --editor openai --timeout 30 --debug
```

Correction eval：

```powershell
python src/main.py eval-corrections --cases eval_cases/correction --editor openai --timeout 60 --debug
python src/main.py eval-corrections --cases eval_cases/correction_en --editor openai --timeout 60 --debug
```

如果暂时没有 API key，可以用 `eval-append --editor rules` 验证基础 CLI 路径。

## 隐私

以下本地文件已被忽略：

- `secrets.local.yaml`
- `.env`
- `.tmp_voice_context/`
- `local_samples/`
- `debug_logs/`
- `eval_runs/`
- `*.wav`, `*.mp3`, `*.m4a`

不要提交 API key、本地录音、transcript、模型文件或私有项目 context。

## 路线图

- 公开版 profile / project memory。
- macOS / Linux 音频和热键适配。
- Windows 安装包。
- 更多 STT API provider 验证。
- 更通用的社区 eval cases。
