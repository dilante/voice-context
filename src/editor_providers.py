from __future__ import annotations

import os
import time
from typing import Any
from urllib.parse import urlparse

try:
    from .utils import PROJECT_ROOT, editor_no_think_enabled, provider_section_config, resolve_editor_provider_config, sanitize_text
except ImportError:
    from utils import PROJECT_ROOT, editor_no_think_enabled, provider_section_config, resolve_editor_provider_config, sanitize_text


class ProviderError(RuntimeError):
    pass


def _supports_no_think_prompt(model: str) -> bool:
    normalized = str(model or "").strip().lower()
    return (
        normalized.startswith("qwen3")
        or "qwen3" in normalized
        or normalized.startswith("qwq")
        or "deepseek-r1" in normalized
    )


def _should_inject_no_think(config: dict[str, Any], model: str, *, provider: str = "") -> bool:
    if not _supports_no_think_prompt(model):
        return False
    if str(provider or "").strip().lower() != "ollama":
        return False
    return editor_no_think_enabled(config)


def _inject_no_think_prefix(user_prompt: str) -> tuple[str, bool]:
    text = str(user_prompt or "")
    if text.lstrip().startswith("/no_think"):
        return text, False
    return f"/no_think\n{text}", True


def _post_chat_completion(
    *,
    config: dict[str, Any],
    base_url: str,
    model: str,
    api_key: str,
    system_prompt: str,
    user_prompt: str,
    temperature: float,
    timeout_sec: float,
    provider: str = "",
    resolved_model: str = "",
    api_key_masked: str = "",
    extra_headers: dict[str, str] | None = None,
    extra_body: dict[str, Any] | None = None,
    api_extra_body_keys: list[str] | None = None,
    log_base_url_host_only: bool = False,
) -> dict[str, Any]:
    try:
        import requests
    except ImportError as exc:
        raise ProviderError("requests 未安装，无法调用模型 provider。") from exc

    no_think_enabled = _should_inject_no_think(config, model, provider=provider)
    no_think_injected = False
    effective_user_prompt = user_prompt
    if no_think_enabled:
        effective_user_prompt, no_think_injected = _inject_no_think_prefix(user_prompt)

    url = base_url.rstrip("/") + "/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    if extra_headers:
        headers.update(extra_headers)
    payload: dict[str, Any] = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": effective_user_prompt},
        ],
        "temperature": temperature,
        "stream": False,
    }
    if extra_body:
        payload.update(extra_body)

    started = time.perf_counter()
    try:
        response = requests.post(url, headers=headers, json=payload, timeout=timeout_sec)
    except requests.Timeout as exc:
        raise ProviderError(f"模型调用超时（>{timeout_sec} 秒）。") from exc
    except requests.ConnectionError as exc:
        raise ProviderError(f"无法连接到模型服务：{base_url}") from exc
    except requests.RequestException as exc:
        raise ProviderError(f"模型请求失败：{exc}") from exc

    try:
        response.raise_for_status()
    except requests.HTTPError as exc:
        response_text = sanitize_text(response.text.strip(), max_chars=300)
        if response_text:
            raise ProviderError(f"模型服务返回 HTTP {response.status_code}：{response_text[:300]}") from exc
        raise ProviderError(f"模型服务返回 HTTP {response.status_code}。") from exc

    try:
        data = response.json()
        message = data["choices"][0]["message"]
        content = message["content"]
    except Exception as exc:
        raise ProviderError("模型返回格式异常，未找到 choices[0].message.content。") from exc

    if not isinstance(content, str) or not content.strip():
        raise ProviderError("模型返回为空。")
    elapsed_ms = int((time.perf_counter() - started) * 1000)
    usage = data.get("usage", {}) if isinstance(data, dict) else {}
    if not isinstance(usage, dict):
        usage = {}
    parsed_url = urlparse(base_url)
    safe_base_url = parsed_url.netloc or base_url
    if not log_base_url_host_only:
        safe_base_url = base_url
    return {
        "content": content.strip(),
        "raw_content": content,
        "message_keys": sorted(message.keys()) if isinstance(message, dict) else [],
        "model": model,
        "resolved_model": resolved_model or model,
        "base_url": safe_base_url,
        "base_url_host": parsed_url.netloc or "",
        "timeout_sec": timeout_sec,
        "status_code": response.status_code,
        "usage": {
            "prompt_tokens": usage.get("prompt_tokens", 0),
            "completion_tokens": usage.get("completion_tokens", 0),
            "total_tokens": usage.get("total_tokens", 0),
        },
        "provider_name": provider,
        "api_key_masked": api_key_masked,
        "api_extra_body_keys": api_extra_body_keys or [],
        "provider_elapsed_ms": elapsed_ms,
        "no_think_enabled": no_think_enabled,
        "qwen3_no_think_enabled": no_think_enabled,
        "no_think_injected": no_think_injected,
    }


def call_ollama(
    *,
    config: dict[str, Any],
    system_prompt: str,
    user_prompt: str,
    temperature: float,
    timeout_sec: float,
) -> dict[str, Any]:
    try:
        resolved = resolve_editor_provider_config(config, provider_name="ollama", project_root=PROJECT_ROOT)
    except ValueError as exc:
        raise ProviderError(str(exc)) from exc
    return _post_chat_completion(
        config=config,
        base_url=str(resolved["base_url"]),
        model=str(resolved["resolved_model"]),
        api_key=str(resolved["api_key"]),
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        temperature=temperature,
        timeout_sec=timeout_sec,
        provider="ollama",
        resolved_model=str(resolved["resolved_model"]),
        api_key_masked=str(resolved["api_key_masked"]),
    )


def call_openai_compatible(
    *,
    config: dict[str, Any],
    system_prompt: str,
    user_prompt: str,
    temperature: float,
    timeout_sec: float,
) -> dict[str, Any]:
    provider = provider_section_config(config, "openai_compatible")
    api_key_env = provider.get("api_key_env", "OPENAI_API_KEY")
    api_key = os.environ.get(api_key_env, "")
    if not api_key:
        raise ProviderError(f"环境变量 {api_key_env} 未设置。")

    return _post_chat_completion(
        config=config,
        base_url=provider.get("base_url", "https://api.openai.com/v1"),
        model=provider.get("model", "your-fast-model"),
        api_key=api_key,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        temperature=temperature,
        timeout_sec=timeout_sec,
        provider="openai_compatible",
    )


def call_api(
    *,
    config: dict[str, Any],
    system_prompt: str,
    user_prompt: str,
    temperature: float,
    timeout_sec: float,
) -> dict[str, Any]:
    try:
        resolved = resolve_editor_provider_config(config, project_root=PROJECT_ROOT)
    except ValueError as exc:
        raise ProviderError(str(exc)) from exc
    if str(resolved.get("kind", "")) == "ollama":
        return _post_chat_completion(
            config=config,
            base_url=str(resolved["base_url"]),
            model=str(resolved["resolved_model"]),
            api_key=str(resolved["api_key"]),
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=temperature,
            timeout_sec=timeout_sec,
            provider="ollama",
            resolved_model=str(resolved["resolved_model"]),
            api_key_masked=str(resolved["api_key_masked"]),
        )
    provider = str(resolved.get("provider", "api"))
    extra_headers: dict[str, str] = {}
    extra_body: dict[str, Any] = {}
    extra_body_keys: list[str] = []
    resolved_model = str(resolved["resolved_model"])
    api_no_think = bool(resolved.get("api_no_think", False))
    if provider == "openrouter":
        extra_headers = {
            "HTTP-Referer": "https://localhost",
            "X-OpenRouter-Title": "voice-context-local",
        }
    if api_no_think and (provider in {"deepseek", "dashscope"} or resolved_model.lower().startswith("qwen3")):
        extra_body["thinking"] = {"type": "disabled"}
        extra_body_keys.append("thinking")
    return _post_chat_completion(
        config=config,
        base_url=str(resolved["base_url"]),
        model=resolved_model,
        api_key=str(resolved["api_key"]),
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        temperature=temperature,
        timeout_sec=timeout_sec,
        provider=provider,
        resolved_model=str(resolved["resolved_model"]),
        api_key_masked=str(resolved["api_key_masked"]),
        extra_headers=extra_headers,
        extra_body=extra_body,
        api_extra_body_keys=extra_body_keys,
        log_base_url_host_only=True,
    )


def warmup_ollama(*, config: dict[str, Any], timeout_sec: float = 2.0) -> dict[str, Any]:
    started = time.perf_counter()
    result = call_ollama(
        config=config,
        system_prompt="你是模型预热器。只回复 ok。",
        user_prompt="ok",
        temperature=0,
        timeout_sec=timeout_sec,
    )
    elapsed_ms = int((time.perf_counter() - started) * 1000)
    return {
        "ok": True,
        "model": result.get("model", ""),
        "base_url": result.get("base_url", ""),
        "elapsed_ms": elapsed_ms,
        "message": f"ollama ready ({result.get('model', '')})",
    }
