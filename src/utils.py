from __future__ import annotations

import json
import os
import re
import subprocess
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_CONFIG: dict[str, Any] = {
    "context": {
        "default_dir": "~/.voice_context",
        "index_file": "index.json",
        "sessions_dir": "sessions",
        "recent_messages_max": 8,
        "max_message_chars": 1200,
    },
    "editor": {
        "provider": "dashscope",
        "timeout_sec": 8.0,
        "temperature": 0,
        "correction_max_repair_attempts": 3,
        "no_think": True,
        "prompts": {
            "append_system": (
                "你是 listen 模式下的新 Draft block 整理器。\n"
                "你只负责把当前这段 raw STT 整理成要追加到 Draft 尾部的新 Draft blocks。\n"
                "只输出用户真正想表达的 Draft 内容。\n"
                "绝不要输出说明文字、指令、字段名、标题、上下文标签、配置值、schema、示例或 prompt 本身。\n"
                "绝不要回显 prompt 里的字段名。\n"
                "不要重写已有 Draft，不要返回完整 Draft，不要解释，不要 JSON，不要 Markdown code block。\n"
                "保留 raw STT 中已经成立的技术术语、UI 标签、命令名、键名、变量名和代码风格词。\n"
                "不要把有效技术词替换成发音相近但无关的词。\n"
                "只有 raw_stt 本身或 term_whitelist 明确支持时，才修正明显 STT 错误；如果不确定，优先保留 raw wording。\n"
                "不要把普通中文短语翻译成英文，也不要把普通英文短语翻译成中文；除非用户明确要求翻译或替换语言。\n"
                "只输出新的 Draft block 文本；如果需要多个 blocks，用空行分隔。"
            ),
            "append_user_template": (
                "你只做 raw STT 的最小清理，用于 append 新 Draft 文本。\n"
                "不要执行命令，不要推断用户想编辑现有 Draft，不要根据其他上下文改写任务。\n"
                "只允许：标点、断句、去口头禅、轻微语法整理、明显 ASR 错词修正。\n"
                "优先做最小转写清理，不要改写成另一种说法，不要在中文字符之间插入空格。\n"
                "不要删除/替换/重排已有 Draft，不要把一句话改写成别的任务。\n"
                "如果 raw STT 里有 delete/remove/change/replace/删除/改成 这类词，必须按字面保留，不要替用户执行。\n"
                "只输出要 append 的新文本；如需多个 blocks，用空行分隔；不要 JSON，不要解释。\n"
                "primary_language={primary_language}\n"
                "secondary_languages={secondary_languages}\n"
                "mixed_language_mode={mixed_language_mode}\n"
                "stt_homophone_hint={stt_homophone_hint}\n\n"
                "term_whitelist={term_bank_hints}\n\n"
                "raw_stt:\n"
                "{raw_stt}"
            ),
            "correction_system": (
                "你是 Draft block 编辑器。\n"
                "用户提供的是修改请求，不是新增内容。\n"
                "只返回 JSON，绝不解释。\n"
                "禁止把编辑说明写进 Draft block。"
            ),
            "correction_repair_system": (
                "You repair a JSON response for a Draft editing pipeline.\n"
                "Return exactly one corrected JSON object.\n"
                "No markdown. No code fence. No explanation."
            ),
            "correction_user_template": (
                "You are a Draft edit ops planner.\n"
                "Return exactly one JSON object.\n"
                "No markdown. No code fence. No explanation.\n"
                "The only top-level keys are changed, reason, ops.\n"
                "The ops are authoritative. The reason is only a short label.\n\n"
                "You will see BLOCKS_TABLE.\n"
                "The number inside [] is the visible position.\n"
                "Use visible positions in every op.\n"
                "Do not output block ids.\n"
                "Do not invent hidden ids.\n"
                "Do not use any position not shown in BLOCKS_TABLE.\n\n"
                "Core rule:\n"
                "If the user refers to a sentence, block, line, item, row, or entry in any language, map it to a visible position from BLOCKS_TABLE.\n"
                "If you cannot confidently map the target to a visible position, return noop.\n\n"
                "Accepted ops:\n"
                "- replace_text: {\"op\":\"replace_text\",\"position\":2,\"old_exact\":\"...\",\"new_text\":\"...\"}\n"
                "- replace_block: {\"op\":\"replace_block\",\"position\":2,\"text\":\"...\"}\n"
                "- delete_blocks: {\"op\":\"delete_blocks\",\"positions\":[2,4]}\n"
                "- swap_blocks: {\"op\":\"swap_blocks\",\"a_position\":3,\"b_position\":4}\n"
                "- move_block: {\"op\":\"move_block\",\"position\":4,\"to\":\"front\"}\n"
                "- move_block: {\"op\":\"move_block\",\"position\":4,\"to\":\"end\"}\n"
                "- move_block: {\"op\":\"move_block\",\"position\":4,\"to\":\"before\",\"ref_position\":1}\n"
                "- move_block: {\"op\":\"move_block\",\"position\":4,\"to\":\"after\",\"ref_position\":1}\n"
                "- noop: {\"op\":\"noop\"}\n\n"
                "Use raw_user_modification_request as the correction request.\n"
                "Do not invent positions.\n"
                "Do not use positions that are not shown in BLOCKS_TABLE.\n\n"
                "Delete:\n"
                "- If the user asks to delete/remove/drop/clear a target, use delete_blocks.\n"
                "- positions is always an array.\n\n"
                "Order:\n"
                "- If the user asks to swap/exchange/switch two blocks, use swap_blocks.\n"
                "- If the user asks to move one block to another position, use move_block.\n"
                "- Do not calculate a full final order list.\n"
                "- If unsure, return noop.\n\n"
                "Replacement:\n"
                "- First map the target to a visible position.\n"
                "- Then extract old_exact and new_text.\n"
                "- old_exact is the smallest exact existing substring to replace.\n"
                "- new_text is only the replacement fragment.\n"
                "- Do not include target locator words in old_exact or new_text.\n"
                "- Do not use the whole block as old_exact unless the user explicitly asks to rewrite or replace the entire block.\n"
                "- If the user asks to change one word or phrase, use replace_text, not replace_block.\n"
                "- If still unsure, return noop.\n"
                "Language rules:\n"
                "- Preserve ordinary phrase language.\n"
                "- Do not translate ordinary Chinese to English.\n"
                "- Do not translate ordinary English to Chinese.\n"
                "- Preserve technical/UI/code/model terms and whitelisted terms.\n\n"
                "No-op rules:\n"
                "- If the requested edit would make no change, return changed=false with noop.\n"
                "- If the target is unclear, return changed=false with noop.\n"
                "- If exact replacement cannot be safely located, return changed=false with noop.\n\n"
                "primary_language: {primary_language}\n"
                "secondary_languages: {secondary_languages}\n"
                "mixed_language_mode: {mixed_language_mode}\n"
                "stt_homophone_hint: {stt_homophone_hint}\n"
                "term_whitelist: {term_bank_hints}\n\n"
                "BLOCKS_TABLE:\n"
                "{blocks_table_text}\n\n"
                "raw_user_modification_request:\n"
                "{raw_correction_text}"
            ),
            "correction_repair_user_template": (
                "Repair the JSON response for this stage.\n"
                "Return exactly one corrected JSON object.\n"
                "No markdown. No code fence. No explanation.\n"
                "Keep the same accepted ops schema.\n"
                "Fix only the validation issue.\n"
                "If unsafe or ambiguous, return {\"changed\":false,\"reason\":\"noop\",\"ops\":[{\"op\":\"noop\"}]}.\n\n"
                "stage:\n"
                "{repair_stage}\n\n"
                "required_contract:\n"
                "{stage_contract_text}\n\n"
                "validation_error:\n"
                "{validation_error}\n\n"
                "raw_user_modification_request:\n"
                "{raw_correction_text}\n\n"
                "BLOCKS_TABLE:\n"
                "{blocks_table_text}\n\n"
                "parsed_candidate_if_any:\n"
                "{parsed_candidate_json}\n\n"
                "previous_llm_response:\n"
                "{previous_llm_response}"
            ),
        },
    },
    "secrets": {
        "file": "secrets.local.yaml",
    },
    "editor_providers": {
        "ollama": {
            "kind": "ollama",
            "base_url": "http://localhost:11434/v1",
            "model": "qwen3:4b",
            "api_key": "ollama",
        },
        "openai": {
            "kind": "openai_compatible",
            "base_url": "https://api.openai.com/v1",
            "model": "gpt-5.2-mini",
            "api_key_name": "openai",
        },
        "openrouter": {
            "kind": "openai_compatible",
            "base_url": "https://openrouter.ai/api/v1",
            "model": "google/gemini-2.5-flash-lite",
            "api_key_name": "openrouter",
        },
        "google": {
            "kind": "openai_compatible",
            "base_url": "https://generativelanguage.googleapis.com/v1beta/openai",
            "model": "gemini-3.5-flash",
            "api_key_name": "google",
        },
        "deepseek": {
            "kind": "openai_compatible",
            "base_url": "https://api.deepseek.com",
            "model": "deepseek-v4-flash",
            "api_key_name": "deepseek",
        },
        "dashscope": {
            "kind": "openai_compatible",
            "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
            "model": "qwen3-max",
            "api_key_name": "dashscope",
        },
    },
    "stt": {
        "provider": "qwen_local",
        "mode": "qwen_local",
    },
    "stt_providers": {
        "qwen_local": {
            "kind": "local_qwen",
            "model_path": "",
            "model_name": "Qwen/Qwen3-ASR-0.6B",
            "device": "auto",
            "dtype": "auto",
            "trust_remote_code": True,
        },
        "openai": {
            "kind": "api_transcription",
            "base_url": "https://api.openai.com/v1",
            "model": "gpt-4o-mini-transcribe",
            "api_key_name": "openai",
        },
        "dashscope": {
            "kind": "api_transcription",
            "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
            "model": "qwen3-asr-flash",
            "api_key_name": "dashscope",
        },
        "google": {
            "kind": "api_audio_model",
            "base_url": "https://generativelanguage.googleapis.com/v1beta/openai",
            "model": "gemini-2.5-flash",
            "api_key_name": "google",
        },
    },
    "audio": {
        "sample_rate": 16000,
        "channels": 1,
        "hotkey": "ctrl+alt+space",
        "hold_to_record": True,
        "temp_dir": "~/.voice_context/tmp_audio",
        "keep_recordings": False,
    },
    "voice_queue": {
        "merge_window_sec": 0,
        "stt_workers": 1,
        "max_segments_per_batch": 0,
        "copy_policy": "latest_only",
        "live_clipboard_update": True,
        "preload_stt_on_start": True,
        "preload_editor_on_start": True,
        "language_context": {
            "primary_language": "zh",
            "secondary_languages": ["en"],
            "stt_homophone_hint": True,
        },
        "command_keywords": {
            "undo": ["撤销", "回退", "undo"],
            "correction_prefixes": ["纠错", "更正", "修正"],
        },
    },
    "debug": {
        "save_replay_logs": False,
        "replay_log_dir": "./debug_logs",
        "eval_run_dir": "./eval_runs",
    },
    "output": {
        "copy_to_clipboard": True,
    },
}
DEFAULT_CONFIG["prompts"] = DEFAULT_CONFIG["editor"].pop("prompts")

SECRET_PATTERNS = [
    (
        re.compile(
            r"(?i)\b(api[_-]?key|token|secret|password|cookie)\b\s*[:=]\s*([^\s,;]+)"
        ),
        r"\1=[REDACTED]",
    ),
    (re.compile(r"(?i)\bBearer\s+[A-Za-z0-9._\-+/=]+"), "Bearer [REDACTED]"),
    (re.compile(r"\bsk-[A-Za-z0-9_\-]{12,}\b"), "[REDACTED_API_KEY]"),
    (re.compile(r"\b[A-Za-z0-9_\-]{24,}\.[A-Za-z0-9_\-]{6,}\.[A-Za-z0-9_\-]{20,}\b"), "[REDACTED_TOKEN]"),
]

API_PROVIDER_PRESETS: dict[str, dict[str, str]] = {
    "openai": {
        "base_url": "https://api.openai.com/v1",
        "default_model": "gpt-5.2-mini",
    },
    "openrouter": {
        "base_url": "https://openrouter.ai/api/v1",
        "default_model": "google/gemini-2.5-flash-lite",
    },
    "google": {
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai",
        "default_model": "gemini-3.5-flash",
    },
    "deepseek": {
        "base_url": "https://api.deepseek.com",
        "default_model": "deepseek-v4-flash",
    },
    "dashscope": {
        "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "default_model": "qwen-plus",
    },
}


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    result = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def load_yaml_file(path: str | Path) -> dict[str, Any]:
    yaml_path = Path(path)
    if not yaml_path.exists():
        return {}
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("pyyaml 未安装，无法读取 YAML 配置。") from exc
    with yaml_path.open("r", encoding="utf-8") as file:
        loaded = yaml.safe_load(file) or {}
    return loaded if isinstance(loaded, dict) else {}


def load_config(config_path: Path | None = None) -> dict[str, Any]:
    path = config_path or PROJECT_ROOT / "config.yaml"
    if not path.exists():
        return deepcopy(DEFAULT_CONFIG)

    loaded = load_yaml_file(path)
    if not loaded:
        return deepcopy(DEFAULT_CONFIG)
    merged = deep_merge(DEFAULT_CONFIG, loaded)
    loaded_editor = loaded.get("editor", {})
    if isinstance(loaded_editor, dict):
        for section_name in ("api", "ollama", "openai_compatible"):
            legacy_section = loaded_editor.get(section_name)
            if isinstance(legacy_section, dict):
                _merge_legacy_editor_provider(merged, section_name, legacy_section)
    for section_name in ("api", "ollama", "openai_compatible"):
        legacy_section = loaded.get(section_name)
        if isinstance(legacy_section, dict):
            _merge_legacy_editor_provider(merged, section_name, legacy_section)
    return merged


def _merge_legacy_editor_provider(config: dict[str, Any], section_name: str, section: dict[str, Any]) -> None:
    providers = config.setdefault("editor_providers", {})
    if not isinstance(providers, dict):
        config["editor_providers"] = {}
        providers = config["editor_providers"]
    if section_name == "ollama":
        providers["ollama"] = deep_merge(providers.get("ollama", {}), {"kind": "ollama", **section})
        return
    if section_name == "openai_compatible":
        providers["openai_compatible"] = deep_merge(
            providers.get("openai_compatible", {}),
            {
                "kind": "openai_compatible",
                "base_url": section.get("base_url", "https://api.openai.com/v1"),
                "model": section.get("model", "your-fast-model"),
                "api_key_env": section.get("api_key_env", "OPENAI_API_KEY"),
            },
        )
        return
    if section_name == "api":
        provider_name = str(section.get("provider", "dashscope") or "dashscope").strip().lower()
        if provider_name == "custom":
            provider_name = "dashscope"
        preset = API_PROVIDER_PRESETS.get(provider_name, {})
        providers[provider_name] = deep_merge(
            providers.get(provider_name, {}),
            {
                "kind": "openai_compatible",
                "base_url": section.get("base_url") or preset.get("base_url", ""),
                "model": section.get("model") or preset.get("default_model", ""),
                "api_key_name": provider_name,
                "no_think": section.get("no_think", section.get("qwen3_no_think", section.get("thinking", ""))),
            },
        )
        if section.get("secrets_file"):
            config.setdefault("secrets", {})
            if isinstance(config["secrets"], dict):
                config["secrets"]["file"] = section.get("secrets_file")


def expand_user_path(value: str) -> Path:
    return Path(os.path.expandvars(os.path.expanduser(value)))


def resolve_relative_path(value: str, project_root: Path = PROJECT_ROOT) -> Path:
    path = expand_user_path(value)
    if path.is_absolute():
        return path
    return project_root / path


def editor_provider_name(config: dict[str, Any]) -> str:
    editor_config = config.get("editor", {}) if isinstance(config, dict) else {}
    if not isinstance(editor_config, dict):
        return "rules"
    provider = str(editor_config.get("provider") or editor_config.get("mode") or "rules").strip().lower()
    if provider == "api":
        legacy_api = provider_section_config(config, "api")
        return str(legacy_api.get("provider", "dashscope") or "dashscope").strip().lower()
    return provider or "rules"


def provider_section_config(config: dict[str, Any], section_name: str) -> dict[str, Any]:
    section = config.get(section_name, {}) if isinstance(config, dict) else {}
    if isinstance(config, dict) and section_name in config and isinstance(section, dict):
        return section
    editor_config = config.get("editor", {}) if isinstance(config, dict) else {}
    if not isinstance(editor_config, dict):
        return {}
    legacy_section = editor_config.get(section_name, {})
    return legacy_section if isinstance(legacy_section, dict) else {}


def editor_provider_names(config: dict[str, Any]) -> list[str]:
    providers = config.get("editor_providers", {}) if isinstance(config, dict) else {}
    if not isinstance(providers, dict):
        return ["rules", "ollama", *sorted(API_PROVIDER_PRESETS.keys())]
    return ["rules", *sorted(str(name) for name in providers.keys())]


def editor_provider_config(config: dict[str, Any], provider_name: str | None = None) -> dict[str, Any]:
    name = (provider_name or editor_provider_name(config)).strip().lower()
    providers = config.get("editor_providers", {}) if isinstance(config, dict) else {}
    if isinstance(providers, dict):
        provider = providers.get(name, {})
        if isinstance(provider, dict) and provider:
            return provider
    if name == "ollama":
        legacy = provider_section_config(config, "ollama")
        return {"kind": "ollama", "base_url": "http://localhost:11434/v1", "model": "qwen3:4b", **legacy}
    if name == "openai_compatible":
        legacy = provider_section_config(config, "openai_compatible")
        return {
            "kind": "openai_compatible",
            "base_url": legacy.get("base_url", "https://api.openai.com/v1"),
            "model": legacy.get("model", "your-fast-model"),
            "api_key_env": legacy.get("api_key_env", "OPENAI_API_KEY"),
        }
    preset = API_PROVIDER_PRESETS.get(name)
    if preset:
        return {
            "kind": "openai_compatible",
            "base_url": preset["base_url"],
            "model": preset["default_model"],
            "api_key_name": name,
        }
    return {}


def editor_no_think_enabled(config: dict[str, Any], provider_config: dict[str, Any] | None = None) -> bool:
    editor_config = config.get("editor", {}) if isinstance(config, dict) else {}
    if not isinstance(editor_config, dict):
        editor_config = {}
    effective_provider_config = provider_config if isinstance(provider_config, dict) else {}
    if "no_think" in effective_provider_config:
        return bool(effective_provider_config.get("no_think"))
    if "qwen3_no_think" in effective_provider_config:
        return bool(effective_provider_config.get("qwen3_no_think"))
    legacy_thinking = effective_provider_config.get("thinking", effective_provider_config.get("deepseek_thinking"))
    if legacy_thinking is not None:
        return str(legacy_thinking).strip().lower() == "disabled"
    if "no_think" in editor_config:
        return bool(editor_config.get("no_think"))
    if "qwen3_no_think" in editor_config:
        return bool(editor_config.get("qwen3_no_think"))
    return True


def editor_api_secrets_path(config: dict[str, Any], project_root: Path = PROJECT_ROOT) -> Path:
    secrets_config = config.get("secrets", {}) if isinstance(config, dict) else {}
    if not isinstance(secrets_config, dict):
        secrets_config = {}
    api_config = provider_section_config(config, "api")
    raw_path = str(
        secrets_config.get("file")
        or api_config.get("secrets_file")
        or "secrets.local.yaml"
    ).strip()
    return resolve_relative_path(raw_path, project_root=project_root)


def load_editor_secrets(config: dict[str, Any], project_root: Path = PROJECT_ROOT) -> dict[str, Any]:
    return load_yaml_file(editor_api_secrets_path(config, project_root=project_root))


def mask_secret(value: str) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    if len(text) <= 8:
        return (text[:2] + "...") if len(text) > 2 else "..."
    prefix = text[:3] if text.startswith("sk-") else text[:2]
    return f"{prefix}...{text[-4:]}"


def git_is_tracked(path: Path, project_root: Path = PROJECT_ROOT) -> bool:
    try:
        query_path = path
        try:
            query_path = path.resolve().relative_to(project_root.resolve())
        except Exception:
            pass
        result = subprocess.run(
            ["git", "ls-files", "--error-unmatch", str(query_path)],
            cwd=str(project_root),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
    except Exception:
        return False
    return result.returncode == 0


def resolve_api_provider_config(config: dict[str, Any], project_root: Path = PROJECT_ROOT) -> dict[str, Any]:
    return resolve_editor_provider_config(config, project_root=project_root)


def resolve_editor_provider_config(
    config: dict[str, Any],
    provider_name: str | None = None,
    project_root: Path = PROJECT_ROOT,
) -> dict[str, Any]:
    provider = (provider_name or editor_provider_name(config)).strip().lower()
    provider_config = editor_provider_config(config, provider)
    if not provider_config:
        supported = ", ".join(editor_provider_names(config))
        raise ValueError(f"不支持的 editor.provider: {provider}。支持：{supported}。")
    kind = str(provider_config.get("kind", "openai_compatible") or "openai_compatible").strip().lower()
    provider_no_think = editor_no_think_enabled(config, provider_config)
    if kind == "ollama":
        model = str(provider_config.get("model", "qwen3:4b") or "qwen3:4b").strip()
        base_url = str(provider_config.get("base_url", "http://localhost:11434/v1") or "http://localhost:11434/v1").strip()
        api_key = str(provider_config.get("api_key", "") or "").strip() or _api_key_from_secrets(config, provider_config, provider, project_root) or "ollama"
        return {
            "provider": provider,
            "kind": kind,
            "base_url": base_url.rstrip("/"),
            "model": model,
            "resolved_model": model,
            "api_key": api_key,
            "api_key_masked": mask_secret(api_key),
            "key_loaded": bool(api_key),
            "secrets_file": str(editor_api_secrets_path(config, project_root=project_root)),
            "secrets_file_tracked": False,
            "api_no_think": provider_no_think,
        }
    if kind == "openai_compatible":
        return _resolve_openai_compatible_provider_config(config, provider, provider_config, project_root)
    raise ValueError(f"不支持的 editor provider kind: {kind}。")


def _api_key_from_secrets(
    config: dict[str, Any],
    provider_config: dict[str, Any],
    provider: str,
    project_root: Path,
) -> str:
    secrets_path = editor_api_secrets_path(config, project_root=project_root)
    secrets = load_editor_secrets(config, project_root=project_root)
    api_keys = secrets.get("api_keys", {}) if isinstance(secrets, dict) else {}
    if not isinstance(api_keys, dict):
        api_keys = {}
    key_name = str(provider_config.get("api_key_name", provider) or provider).strip()
    return str(api_keys.get(key_name, "") or "").strip()


def _resolve_openai_compatible_provider_config(
    config: dict[str, Any],
    provider: str,
    provider_config: dict[str, Any],
    project_root: Path,
) -> dict[str, Any]:
    provider_no_think = editor_no_think_enabled(config, provider_config)
    secrets_path = editor_api_secrets_path(config, project_root=project_root)
    model = str(provider_config.get("model", "") or "").strip()
    base_url = str(provider_config.get("base_url", "") or "").strip()
    api_key = _api_key_from_secrets(config, provider_config, provider, project_root)
    api_key_env = str(provider_config.get("api_key_env", "") or "").strip()
    if not api_key and api_key_env:
        api_key = str(os.environ.get(api_key_env, "") or "").strip()
    if not secrets_path.exists():
        key_name = str(provider_config.get("api_key_name", provider) or provider)
        raise ValueError(f"API secrets 文件不存在：{secrets_path}。请复制 secrets.example.yaml 为 secrets.local.yaml 并填写 api_keys.{key_name}。")
    if not api_key:
        key_name = str(provider_config.get("api_key_name", provider) or provider)
        if api_key_env:
            raise ValueError(f"缺少 API key：请在 {secrets_path} 填写 api_keys.{key_name}，或设置环境变量 {api_key_env}。")
        raise ValueError(f"缺少 API key：请在 {secrets_path} 填写 api_keys.{key_name}。")
    if not base_url:
        raise ValueError(f"editor_providers.{provider}.base_url 不能为空。")
    if not model:
        raise ValueError(f"editor_providers.{provider}.model 不能为空。")

    return {
        "provider": provider,
        "kind": "openai_compatible",
        "base_url": base_url.rstrip("/"),
        "model": model,
        "resolved_model": model,
        "api_key": api_key,
        "api_key_masked": mask_secret(api_key),
        "key_loaded": bool(api_key),
        "secrets_file": str(secrets_path),
        "secrets_file_tracked": git_is_tracked(secrets_path, project_root=project_root),
        "api_no_think": provider_no_think,
    }


def editor_status(config: dict[str, Any], project_root: Path = PROJECT_ROOT) -> dict[str, Any]:
    provider = editor_provider_name(config)
    provider_config = editor_provider_config(config, provider)
    status = {
        "provider": provider,
        "kind": str(provider_config.get("kind", "") or ""),
        "model": str(provider_config.get("model", "") or ""),
        "resolved_model": "",
        "base_url": str(provider_config.get("base_url", "") or ""),
        "secrets_file": str(editor_api_secrets_path(config, project_root=project_root)),
        "key_loaded": False,
        "key_masked": "",
        "editor_no_think": editor_no_think_enabled(config),
        "provider_no_think": editor_no_think_enabled(config, provider_config),
        "warning": "",
    }
    if provider == "rules":
        status.update({"kind": "rules", "resolved_model": ""})
        return status
    try:
        resolved = resolve_editor_provider_config(config, provider_name=provider, project_root=project_root)
        status.update(
            {
                "kind": resolved.get("kind", status["kind"]),
                "model": resolved.get("model", status["model"]),
                "resolved_model": resolved.get("resolved_model", ""),
                "base_url": resolved.get("base_url", status["base_url"]),
                "key_loaded": bool(resolved.get("key_loaded", False)),
                "key_masked": resolved.get("api_key_masked", ""),
                "provider_no_think": bool(resolved.get("api_no_think", status["provider_no_think"])),
                "warning": "secrets file is tracked by git" if resolved.get("secrets_file_tracked") else "",
            }
        )
    except ValueError as exc:
        status["warning"] = sanitize_text(str(exc), max_chars=240)
    return status


def truncate_text(text: str, max_chars: int = 1200) -> str:
    normalized = str(text).replace("\x00", "").strip()
    if len(normalized) <= max_chars:
        return normalized
    return normalized[: max_chars - 20].rstrip() + " ...[TRUNCATED]"


def sanitize_text(text: Any, max_chars: int = 1200) -> str:
    sanitized = truncate_text("" if text is None else str(text), max_chars=max_chars)
    for pattern, replacement in SECRET_PATTERNS:
        sanitized = pattern.sub(replacement, sanitized)
    return sanitized


def sanitize_value(value: Any, max_chars: int = 1200) -> Any:
    if value is None:
        return ""
    if isinstance(value, str):
        return sanitize_text(value, max_chars=max_chars)
    if isinstance(value, list):
        return [sanitize_value(item, max_chars=300) for item in value[:20]]
    if isinstance(value, tuple):
        return [sanitize_value(item, max_chars=300) for item in value[:20]]
    if isinstance(value, dict):
        return {
            sanitize_text(key, max_chars=80): sanitize_value(item, max_chars=max_chars)
            for key, item in value.items()
        }
    return sanitize_text(value, max_chars=max_chars)


def json_dumps(data: Any) -> str:
    return json.dumps(data, ensure_ascii=False, indent=2)


def copy_to_clipboard(text: str) -> tuple[bool, str]:
    try:
        import pyperclip
    except ImportError:
        return False, "pyperclip 未安装，已跳过复制到剪贴板。"

    try:
        pyperclip.copy(text)
    except Exception as exc:
        return False, f"复制到剪贴板失败：{exc}"
    return True, "已复制到剪贴板。"


def replay_log_dir(config: dict[str, Any], *, kind: str = "debug") -> Path:
    debug_config = config.get("debug", {}) if isinstance(config, dict) else {}
    if not isinstance(debug_config, dict):
        debug_config = {}
    if kind == "eval":
        raw_value = str(debug_config.get("eval_run_dir", "~/.voice_context/eval_runs")).strip()
    else:
        raw_value = str(debug_config.get("replay_log_dir", "~/.voice_context/debug_logs")).strip()
    return expand_user_path(raw_value or "~/.voice_context/debug_logs")


def replay_log_path(config: dict[str, Any], *, kind: str = "debug", timestamp: datetime | None = None) -> Path:
    now = timestamp or datetime.now()
    root = replay_log_dir(config, kind=kind)
    if kind == "eval":
        return root / f"{now.strftime('%Y-%m-%d-%H%M%S')}.jsonl"
    return root / f"{now.strftime('%Y-%m-%d')}.jsonl"


def save_replay_log_entry(
    config: dict[str, Any],
    entry: dict[str, Any],
    *,
    kind: str = "debug",
    explicit_path: Path | None = None,
) -> tuple[bool, str]:
    try:
        path = explicit_path or replay_log_path(config, kind=kind)
        path.parent.mkdir(parents=True, exist_ok=True)
        safe_entry = sanitize_value(entry, max_chars=12000)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(safe_entry, ensure_ascii=False) + "\n")
        return True, str(path)
    except Exception as exc:
        return False, sanitize_text(str(exc), max_chars=200)
