from __future__ import annotations

import difflib
import importlib.util
import json
import logging
import os
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any

try:
    from .utils import PROJECT_ROOT, sanitize_text, sanitize_value, load_config
except ImportError:
    from utils import PROJECT_ROOT, sanitize_text, sanitize_value, load_config


TRANSCRIPT_KEYS = ("text", "transcript", "result", "transcription")
QWEN_LOCAL_CONFIG_ERROR = "qwen_local 尚未配置 model_path 或 model_name。"
QWEN_LOCAL_DEPENDENCY_ERROR = (
    "qwen_local 缺少依赖。请安装 `qwen-asr`、`torch`、`transformers`、`accelerate`；如果你使用 ModelScope 版本，也请安装 `modelscope`。"
)


class STTError(RuntimeError):
    pass


class _ScriptModuleFallback(RuntimeError):
    pass


_QWEN_ASR_CACHE: dict[tuple[str, str, str], Any] = {}
_QWEN_ASR_CACHE_LOCK = threading.Lock()


def _quiet_hf_runtime() -> None:
    os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
    for logger_name in ["transformers", "modelscope", "torch"]:
        logging.getLogger(logger_name).setLevel(logging.ERROR)
    try:
        from transformers.utils import logging as transformers_logging  # type: ignore

        transformers_logging.set_verbosity_error()
    except Exception:
        pass


def _resolve_audio_path(audio_path: str) -> Path:
    path = Path(audio_path).expanduser()
    if not path.is_absolute():
        path = Path.cwd() / path
    resolved = path.resolve()
    if not resolved.exists():
        raise STTError(f"音频文件不存在：{resolved}")
    if not resolved.is_file():
        raise STTError(f"音频路径不是文件：{resolved}")
    return resolved


def _normalize_transcript_value(value: Any) -> str:
    if isinstance(value, dict):
        for key in TRANSCRIPT_KEYS:
            item = value.get(key)
            if item is not None and str(item).strip():
                return str(item).strip()
        return sanitize_text(value, max_chars=12000).strip()
    if isinstance(value, list):
        parts: list[str] = []
        for item in value:
            text = _normalize_transcript_value(item)
            if text:
                parts.append(text)
        return " ".join(parts).strip()
    if hasattr(value, "text"):
        text_value = getattr(value, "text")
        if text_value is not None and str(text_value).strip():
            return str(text_value).strip()
    return sanitize_text(value, max_chars=12000).strip()


def _extract_transcript_from_output(stdout_text: str) -> str:
    cleaned = stdout_text.strip()
    if not cleaned:
        return ""

    try:
        payload = json.loads(cleaned)
    except json.JSONDecodeError:
        return cleaned

    return _normalize_transcript_value(payload)


def _detect_expected_transcript(audio_path: Path) -> dict[str, str]:
    expected_path = audio_path.with_suffix(".expected.txt")
    if not expected_path.exists() or not expected_path.is_file():
        return {"expected_transcript_path": "", "expected_transcript_preview": "", "comparison_note": ""}

    expected_text = sanitize_text(expected_path.read_text(encoding="utf-8").strip(), max_chars=12000)
    return {
        "expected_transcript_path": str(expected_path),
        "expected_transcript_preview": sanitize_text(expected_text, max_chars=500),
        "comparison_note": "",
    }


def _comparison_note(expected_text: str, actual_text: str) -> str:
    if not expected_text or not actual_text:
        return ""
    ratio = difflib.SequenceMatcher(a=expected_text, b=actual_text).ratio()
    if ratio < 0.65:
        return "STT 输出与 expected transcript 差异较大，请人工检查。"
    if ratio < 0.9:
        return "STT 输出与 expected transcript 有一定差异，建议人工检查。"
    return "STT 输出与 expected transcript 接近，仍建议人工确认。"


def _build_result(
    *,
    provider: str,
    audio_path: Path,
    text: str,
    asr_context: str,
    started: float,
    config_preview: dict[str, Any] | None = None,
) -> dict[str, Any]:
    transcript = sanitize_text(text, max_chars=12000).strip()
    if not transcript:
        raise STTError(f"{provider} 返回为空。")

    elapsed_ms = int((time.perf_counter() - started) * 1000)
    expected_info = _detect_expected_transcript(audio_path)
    expected_preview = expected_info.get("expected_transcript_preview", "")
    expected_full = ""
    if expected_info.get("expected_transcript_path"):
        expected_full = sanitize_text(
            Path(expected_info["expected_transcript_path"]).read_text(encoding="utf-8").strip(),
            max_chars=12000,
        )

    return {
        "text": transcript,
        "audio_path": str(audio_path),
        "stt_elapsed_ms": elapsed_ms,
        "asr_context_preview": sanitize_text(asr_context, max_chars=500),
        "raw_stt_preview": sanitize_text(transcript, max_chars=500),
        "provider": provider,
        "expected_transcript_path": expected_info.get("expected_transcript_path", ""),
        "expected_transcript_preview": expected_preview,
        "comparison_note": _comparison_note(expected_full, transcript) if expected_full else "",
        "config": sanitize_value(config_preview or {}, max_chars=300),
    }


def _get_stt_mode(config: dict[str, Any]) -> str:
    stt_config = config.get("stt", {})
    if not isinstance(stt_config, dict):
        stt_config = {}
    return str(stt_config.get("provider") or stt_config.get("mode") or "qwen_local").strip().lower() or "qwen_local"


def _get_qwen_local_config(config: dict[str, Any]) -> dict[str, Any]:
    stt_config = config.get("stt", {})
    if not isinstance(stt_config, dict):
        stt_config = {}
    stt_providers = config.get("stt_providers", {})
    if not isinstance(stt_providers, dict):
        stt_providers = {}
    provider_config = stt_providers.get("qwen_local", {})
    legacy_config = stt_config.get("qwen_local", {})
    merged: dict[str, Any] = {}
    if isinstance(provider_config, dict):
        merged.update(provider_config)
    if isinstance(legacy_config, dict):
        merged.update(legacy_config)
    return merged


def _primary_language_from_context(config: dict[str, Any]) -> str:
    queue_config = config.get("voice_queue", {})
    if not isinstance(queue_config, dict):
        queue_config = {}
    language_context = queue_config.get("language_context", {})
    if not isinstance(language_context, dict):
        language_context = {}
    return sanitize_text(language_context.get("primary_language", "zh"), max_chars=40).strip() or "zh"


def _qwen_local_language_setting(config: dict[str, Any], provider_config: dict[str, Any]) -> str:
    if "language" in provider_config and str(provider_config.get("language", "")).strip():
        return sanitize_text(provider_config.get("language", ""), max_chars=40) or _primary_language_from_context(config)
    return _primary_language_from_context(config)


def _get_legacy_script_config(config: dict[str, Any]) -> dict[str, Any]:
    stt_config = config.get("stt", {})
    provider_config = stt_config.get("legacy_script", {})
    if isinstance(provider_config, dict) and provider_config:
        return provider_config

    # Backward compatibility with the older stt.qwen block.
    legacy_qwen = stt_config.get("qwen", {})
    if isinstance(legacy_qwen, dict):
        return {
            "script_path": legacy_qwen.get("script_path", ""),
            "model": legacy_qwen.get("model", ""),
            "language": legacy_qwen.get("language", "auto"),
        }
    return {}


def _get_dummy_config(config: dict[str, Any]) -> dict[str, Any]:
    provider_config = config.get("stt", {}).get("dummy", {})
    return provider_config if isinstance(provider_config, dict) else {}


def _should_use_asr_context(config: dict[str, Any]) -> bool:
    stt_config = config.get("stt", {})
    mode = _get_stt_mode(config)
    if mode == "qwen_local":
        return bool(_get_qwen_local_config(config).get("use_asr_context", True))
    if mode in {"legacy_script", "qwen"}:
        return bool(stt_config.get("use_asr_context", True))
    return True


def should_use_asr_context(config: dict[str, Any]) -> bool:
    return _should_use_asr_context(config)


def _resolve_script_path(script_path: str) -> Path:
    path = Path(script_path).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    resolved = path.resolve()
    if not resolved.exists():
        raise STTError(f"legacy_script 的 script_path 不存在：{resolved}")
    if not resolved.is_file():
        raise STTError(f"legacy_script 的 script_path 不是文件：{resolved}")
    return resolved


def _load_script_module(script_path: Path) -> Any:
    spec = importlib.util.spec_from_file_location("voice_context_qwen_legacy_script", script_path)
    if spec is None or spec.loader is None:
        raise STTError(f"无法加载 legacy_script 模块：{script_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _call_legacy_module_transcribe(script_path: Path, audio_path: Path, asr_context: str) -> str:
    module = _load_script_module(script_path)
    transcribe_fn = getattr(module, "transcribe", None)
    if not callable(transcribe_fn):
        raise _ScriptModuleFallback(f"{script_path.name} 未暴露 transcribe(audio_path, asr_context)")

    try:
        result = transcribe_fn(str(audio_path), asr_context)
    except TypeError as exc:
        raise STTError(
            "已加载 legacy_script，但 transcribe 调用失败。请确认脚本提供 transcribe(audio_path, asr_context)。"
        ) from exc
    except Exception as exc:
        raise STTError(f"legacy_script 模块调用失败：{sanitize_text(exc, max_chars=300)}") from exc

    text = _normalize_transcript_value(result)
    if not text:
        raise STTError("legacy_script 模块返回为空。")
    return text


def _call_legacy_script_subprocess(
    script_path: Path,
    *,
    audio_path: Path,
    asr_context: str,
    model: str,
    language: str,
) -> str:
    command = [str(script_path)]
    if script_path.suffix.lower() == ".py":
        command = [sys.executable, str(script_path)]

    command.extend(["--audio", str(audio_path)])
    if asr_context:
        command.extend(["--asr-context", asr_context])
    if model:
        command.extend(["--model", model])
    if language:
        command.extend(["--language", language])

    env = dict(os.environ)
    env["VOICE_CONTEXT_STT_AUDIO_PATH"] = str(audio_path)
    env["VOICE_CONTEXT_STT_ASR_CONTEXT"] = asr_context
    env["VOICE_CONTEXT_STT_MODEL"] = model
    env["VOICE_CONTEXT_STT_LANGUAGE"] = language

    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
            env=env,
        )
    except OSError as exc:
        raise STTError(f"无法启动 legacy_script：{sanitize_text(exc, max_chars=300)}") from exc

    stdout_text = result.stdout.strip()
    stderr_text = sanitize_text(result.stderr, max_chars=500)
    if result.returncode != 0:
        error_detail = stderr_text or "脚本未返回 stderr。"
        raise STTError(f"legacy_script 执行失败（exit={result.returncode}）：{error_detail}")

    transcript = _extract_transcript_from_output(stdout_text)
    transcript = sanitize_text(transcript, max_chars=12000).strip()
    if not transcript:
        if stderr_text:
            raise STTError(f"legacy_script 未输出有效转写结果：{stderr_text}")
        raise STTError("legacy_script 未输出有效转写结果。")
    return transcript


def _transcribe_legacy_script(audio_path: Path, asr_context: str, config: dict[str, Any]) -> dict[str, Any]:
    legacy_config = _get_legacy_script_config(config)
    script_path_value = str(legacy_config.get("script_path", "")).strip()
    if not script_path_value:
        raise STTError("legacy_script 尚未配置 script_path。")

    model = sanitize_text(legacy_config.get("model", ""), max_chars=120)
    language = sanitize_text(legacy_config.get("language", "auto"), max_chars=40) or "auto"
    script_path = _resolve_script_path(script_path_value)
    started = time.perf_counter()

    if script_path.suffix.lower() == ".py":
        try:
            transcript = _call_legacy_module_transcribe(script_path, audio_path, asr_context)
        except _ScriptModuleFallback:
            transcript = _call_legacy_script_subprocess(
                script_path,
                audio_path=audio_path,
                asr_context=asr_context,
                model=model,
                language=language,
            )
    else:
        transcript = _call_legacy_script_subprocess(
            script_path,
            audio_path=audio_path,
            asr_context=asr_context,
            model=model,
            language=language,
        )

    return _build_result(
        provider="legacy_script",
        audio_path=audio_path,
        text=transcript,
        asr_context=asr_context,
        started=started,
        config_preview={
            "script_path": str(script_path),
            "model": model,
            "language": language,
        },
    )


def _resolve_qwen_local_source(provider_config: dict[str, Any]) -> str:
    model_path = str(provider_config.get("model_path", "")).strip()
    model_name = str(provider_config.get("model_name", "")).strip()
    if model_path:
        path = Path(model_path).expanduser()
        if not path.is_absolute():
            path = PROJECT_ROOT / path
        return str(path.resolve())
    if model_name:
        return model_name
    raise STTError(QWEN_LOCAL_CONFIG_ERROR)


def _resolve_qwen_local_language(language: str) -> str | None:
    value = (language or "").strip()
    if not value or value.lower() in {"auto", "none", "null"}:
        return None
    aliases = {
        "zh": "Chinese",
        "cn": "Chinese",
        "chinese": "Chinese",
        "en": "English",
        "english": "English",
        "yue": "Cantonese",
        "cantonese": "Cantonese",
        "ja": "Japanese",
        "japanese": "Japanese",
        "ko": "Korean",
        "korean": "Korean",
    }
    return aliases.get(value.lower(), value)


def _resolve_qwen_local_device(requested: str, torch: Any) -> str:
    value = (requested or "").strip().lower()
    if value and value != "auto":
        return value
    return "cuda:0" if torch.cuda.is_available() else "cpu"


def _resolve_qwen_local_dtype(requested: str, device: str, torch: Any) -> Any:
    value = (requested or "").strip().lower()
    if value and value != "auto":
        mapping = {
            "float32": torch.float32,
            "fp32": torch.float32,
            "float16": torch.float16,
            "fp16": torch.float16,
            "bfloat16": torch.bfloat16,
            "bf16": torch.bfloat16,
        }
        if value not in mapping:
            raise STTError(f"qwen_local 不支持的 dtype: {requested}")
        return mapping[value]

    if str(device).startswith("cuda"):
        if hasattr(torch.cuda, "is_bf16_supported") and torch.cuda.is_bf16_supported():
            return torch.bfloat16
        return torch.float16
    return torch.float32


def _try_qwen_asr_qwen_local(
    model_source: str,
    *,
    device: str,
    dtype: str,
    language: str,
) -> tuple[Any, str, str]:
    try:
        import torch
        from qwen_asr import Qwen3ASRModel
    except ImportError as exc:
        raise STTError(QWEN_LOCAL_DEPENDENCY_ERROR) from exc

    resolved_device = _resolve_qwen_local_device(device, torch)
    resolved_dtype = _resolve_qwen_local_dtype(dtype, resolved_device, torch)
    cache_key = (model_source, resolved_device, str(resolved_dtype))

    _quiet_hf_runtime()
    with _QWEN_ASR_CACHE_LOCK:
        model = _QWEN_ASR_CACHE.get(cache_key)
        if model is None:
            try:
                model = Qwen3ASRModel.from_pretrained(
                    model_source,
                    dtype=resolved_dtype,
                    device_map=resolved_device,
                )
            except Exception as exc:
                raise STTError(f"qwen_local(qwen_asr) 模型加载失败：{sanitize_text(exc, max_chars=300)}") from exc
            _QWEN_ASR_CACHE[cache_key] = model

    return model, resolved_device, str(resolved_dtype)


def _transcribe_with_qwen_asr_model(model: Any, *, audio_path: Path, language: str) -> str:
    _quiet_hf_runtime()
    try:
        result = model.transcribe(
            audio=str(audio_path),
            language=_resolve_qwen_local_language(language),
            return_time_stamps=False,
        )
    except Exception as exc:
        raise STTError(f"qwen_local(qwen_asr) 转写失败：{sanitize_text(exc, max_chars=300)}") from exc

    text = _normalize_transcript_value(result)
    if not text:
        raise STTError("qwen_local(qwen_asr) 返回为空。")
    return text


def _try_modelscope_qwen_local(
    model_source: str,
    *,
    audio_path: Path,
    language: str,
    asr_context: str,
    trust_remote_code: bool,
) -> str:
    _quiet_hf_runtime()
    try:
        from modelscope.pipelines import pipeline as ms_pipeline
        from modelscope.utils.constant import Tasks
    except ImportError as exc:
        raise STTError(QWEN_LOCAL_DEPENDENCY_ERROR) from exc

    pipeline_kwargs: dict[str, Any] = {"task": Tasks.auto_speech_recognition, "model": model_source}
    if trust_remote_code:
        pipeline_kwargs["trust_remote_code"] = True

    recognizer = ms_pipeline(**pipeline_kwargs)
    call_kwargs: dict[str, Any] = {}
    if language and language != "auto":
        call_kwargs["language"] = language
    if asr_context:
        call_kwargs["prompt"] = asr_context

    try:
        result = recognizer(str(audio_path), **call_kwargs)
    except TypeError:
        result = recognizer(str(audio_path))
    except Exception as exc:
        raise STTError(f"qwen_local(ModelScope) 调用失败：{sanitize_text(exc, max_chars=300)}") from exc

    text = _normalize_transcript_value(result)
    if not text:
        raise STTError("qwen_local(ModelScope) 返回为空。")
    return text


def _try_transformers_qwen_local(
    model_source: str,
    *,
    audio_path: Path,
    device: str,
    language: str,
    asr_context: str,
    trust_remote_code: bool,
) -> str:
    _quiet_hf_runtime()
    try:
        import torch
        from transformers import pipeline
    except ImportError as exc:
        raise STTError(QWEN_LOCAL_DEPENDENCY_ERROR) from exc

    use_cuda = device == "cuda" or (device == "auto" and torch.cuda.is_available())
    if device not in {"auto", "cuda", "cpu"} and device != "":
        use_cuda = device.startswith("cuda")

    pipeline_kwargs: dict[str, Any] = {
        "task": "automatic-speech-recognition",
        "model": model_source,
        "trust_remote_code": trust_remote_code,
    }
    pipeline_kwargs["device"] = 0 if use_cuda else -1
    pipeline_kwargs["torch_dtype"] = torch.float16 if use_cuda else torch.float32

    try:
        recognizer = pipeline(**pipeline_kwargs)
    except Exception as exc:
        raise STTError(f"qwen_local(transformers) 模型加载失败：{sanitize_text(exc, max_chars=300)}") from exc
    generate_kwargs: dict[str, Any] = {}
    if language and language != "auto":
        generate_kwargs["language"] = language
    if asr_context:
        generate_kwargs["prompt"] = asr_context

    try:
        if generate_kwargs:
            result = recognizer(str(audio_path), generate_kwargs=generate_kwargs)
        else:
            result = recognizer(str(audio_path))
    except TypeError:
        result = recognizer(str(audio_path))
    except Exception as exc:
        raise STTError(f"qwen_local(transformers) 调用失败：{sanitize_text(exc, max_chars=300)}") from exc

    text = _normalize_transcript_value(result)
    if not text:
        raise STTError("qwen_local(transformers) 返回为空。")
    return text


def warmup_stt_provider(*, config: dict[str, Any] | None = None) -> dict[str, Any]:
    loaded_config = config or load_config()
    mode = _get_stt_mode(loaded_config)
    started = time.perf_counter()
    if mode != "qwen_local":
        return {
            "ok": True,
            "mode": mode,
            "message": f"warmup skipped for stt.provider={mode}",
            "elapsed_ms": int((time.perf_counter() - started) * 1000),
        }

    provider_config = _get_qwen_local_config(loaded_config)
    model_source = _resolve_qwen_local_source(provider_config)
    device = sanitize_text(provider_config.get("device", "auto"), max_chars=40) or "auto"
    dtype = sanitize_text(provider_config.get("dtype", "auto"), max_chars=40) or "auto"
    language = _qwen_local_language_setting(loaded_config, provider_config)
    try:
        model, resolved_device, resolved_dtype = _try_qwen_asr_qwen_local(
            model_source,
            device=device,
            dtype=dtype,
            language=language,
        )
    except STTError as exc:
        raise STTError(f"stt warmup failed: {exc}") from exc

    return {
        "ok": True,
        "mode": mode,
        "message": f"qwen_local ready on {resolved_device}/{resolved_dtype}",
        "elapsed_ms": int((time.perf_counter() - started) * 1000),
    }


def _transcribe_qwen_local(audio_path: Path, asr_context: str, config: dict[str, Any]) -> dict[str, Any]:
    provider_config = _get_qwen_local_config(config)
    model_source = _resolve_qwen_local_source(provider_config)
    device = sanitize_text(provider_config.get("device", "auto"), max_chars=40) or "auto"
    dtype = sanitize_text(provider_config.get("dtype", "auto"), max_chars=40) or "auto"
    language = _qwen_local_language_setting(config, provider_config)
    trust_remote_code = bool(provider_config.get("trust_remote_code", True))
    started = time.perf_counter()

    errors: list[str] = []
    transcript = ""
    resolved_backend = ""
    resolved_device = device
    resolved_dtype = dtype
    try:
        model, resolved_device, resolved_dtype = _try_qwen_asr_qwen_local(
            model_source,
            device=device,
            dtype=dtype,
            language=language,
        )
        transcript = _transcribe_with_qwen_asr_model(
            model,
            audio_path=audio_path,
            language=language,
        )
        resolved_backend = "qwen_asr"
    except STTError as exc:
        errors.append(str(exc))

    if not transcript:
        try:
            transcript = _try_modelscope_qwen_local(
                model_source,
                audio_path=audio_path,
                language=language,
                asr_context=asr_context,
                trust_remote_code=trust_remote_code,
            )
            resolved_backend = "modelscope"
        except STTError as exc:
            errors.append(str(exc))

    if not transcript:
        try:
            transcript = _try_transformers_qwen_local(
                model_source,
                audio_path=audio_path,
                device=device,
                language=language,
                asr_context=asr_context,
                trust_remote_code=trust_remote_code,
            )
            resolved_backend = "transformers"
        except STTError as exc:
            errors.append(str(exc))

    if not transcript:
        details = " | ".join(errors) if errors else "未获得任何 backend 错误信息。"
        raise STTError(f"qwen_local 调用失败：{details}")

    return _build_result(
        provider="qwen_local",
        audio_path=audio_path,
        text=transcript,
        asr_context=asr_context,
        started=started,
        config_preview={
            "model_source": model_source,
            "backend": resolved_backend,
            "device": device,
            "resolved_device": resolved_device,
            "dtype": dtype,
            "resolved_dtype": resolved_dtype,
            "language": language,
            "trust_remote_code": trust_remote_code,
        },
    )


def _transcribe_dummy(audio_path: Path, asr_context: str, config: dict[str, Any]) -> dict[str, Any]:
    provider_config = _get_dummy_config(config)
    text = sanitize_text(provider_config.get("text", ""), max_chars=12000).strip()
    if not text:
        raise STTError("dummy provider 尚未配置 text。")
    started = time.perf_counter()
    return _build_result(
        provider="dummy",
        audio_path=audio_path,
        text=text,
        asr_context=asr_context,
        started=started,
        config_preview={"text": text},
    )


def transcribe_audio_with_meta(
    audio_path: str,
    asr_context: str = "",
    *,
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    loaded_config = config or load_config()
    resolved_audio_path = _resolve_audio_path(audio_path)
    compact_context = sanitize_text(asr_context if _should_use_asr_context(loaded_config) else "", max_chars=800)
    mode = _get_stt_mode(loaded_config)

    if mode == "qwen_local":
        result = _transcribe_qwen_local(resolved_audio_path, compact_context, loaded_config)
    elif mode == "legacy_script":
        result = _transcribe_legacy_script(resolved_audio_path, compact_context, loaded_config)
    elif mode == "dummy":
        result = _transcribe_dummy(resolved_audio_path, compact_context, loaded_config)
    else:
        raise STTError(f"不支持的 stt.provider: {mode}")

    result["mode"] = mode
    return result
