from __future__ import annotations

import hashlib
import json
import os
import re
from copy import deepcopy
from pathlib import Path
from typing import Any

try:
    from .utils import (
        expand_user_path,
        json_dumps,
        load_config,
        sanitize_text,
        sanitize_value,
        utc_now_iso,
    )
except ImportError:
    from utils import (
        expand_user_path,
        json_dumps,
        load_config,
        sanitize_text,
        sanitize_value,
        utc_now_iso,
    )


CONTEXT_FIELDS = [
    "current_goal",
    "latest_dynamic",
    "recent_ai_summary",
    "recent_user_intent",
    "important_terms",
    "possible_files",
    "constraints",
    "next_likely_commands",
    "updated_at",
]
TERM_BANK_FIELDS = [
    "domain_terms",
    "files",
    "functions",
    "classes",
    "variables",
    "commands",
    "ui_terms",
]

LIST_FIELDS = {
    "important_terms",
    "possible_files",
    "constraints",
    "next_likely_commands",
}
TERM_BANK_LIMIT = 30
DEFAULT_PROFILE_ID = "default-coding"

DEFAULT_SUMMARY: dict[str, Any] = {
    "current_goal": "",
    "latest_dynamic": "",
    "recent_ai_summary": "",
    "recent_user_intent": "",
    "important_terms": [],
    "possible_files": [],
    "constraints": [],
    "next_likely_commands": [],
    "updated_at": "",
}
DEFAULT_TERM_BANK: dict[str, Any] = {
    "domain_terms": [],
    "files": [],
    "functions": [],
    "classes": [],
    "variables": [],
    "commands": [],
    "ui_terms": [],
}

DEFAULT_INDEX: dict[str, Any] = {
    "active_context_id": "",
    "active_session_id": "",
    "active_profile_id": "",
    "contexts": [],
    "profiles": [],
}

MESSAGE_CODE_BLOCK_PATTERN = re.compile(r"```[\s\S]*?```")
LIKELY_CODE_PATTERN = re.compile(r"(?m)^\s{2,}|\b(def|class|function|return|import|from)\b|[{};=<>]{2,}")
SUMMARY_UPDATE_FIELDS = {
    "current_goal",
    "latest_dynamic",
    "recent_ai_summary",
    "recent_user_intent",
    "important_terms",
    "possible_files",
    "constraints",
    "next_likely_commands",
}
JSON_ENVELOPE_KEYS = {
    "payload_json",
    "profile_id",
    "session_id",
    "context_id",
    "project_root",
    "app",
    "agent",
}


def empty_context() -> dict[str, Any]:
    return deepcopy(DEFAULT_SUMMARY)


def empty_term_bank() -> dict[str, Any]:
    return deepcopy(DEFAULT_TERM_BANK)


def empty_index() -> dict[str, Any]:
    return deepcopy(DEFAULT_INDEX)


def normalize_project_root_value(project_root: str | Path | None) -> str:
    raw_path = Path(project_root or os.getcwd()).expanduser()
    try:
        resolved = raw_path.resolve()
    except OSError:
        resolved = raw_path.absolute()
    normalized = resolved.as_posix().rstrip("/")
    if os.name == "nt":
        normalized = normalized.lower()
    return normalized


def build_fallback_context_id(normalized_project_root: str) -> str:
    return hashlib.sha1(normalized_project_root.encode("utf-8")).hexdigest()[:12]


def resolve_context_home(config: dict[str, Any] | None = None) -> Path:
    loaded_config = config or load_config()
    context_config = loaded_config.get("context", {})
    env_home = os.environ.get("VOICE_CONTEXT_HOME", "").strip()
    configured_home = str(context_config.get("default_dir", "~/.voice_context")).strip()
    return expand_user_path(env_home or configured_home or "~/.voice_context")


def _context_paths(config: dict[str, Any] | None = None) -> dict[str, Path]:
    loaded_config = config or load_config()
    context_config = loaded_config.get("context", {})
    context_home = resolve_context_home(loaded_config)
    sessions_dir = context_home / str(context_config.get("sessions_dir", "sessions"))
    return {
        "context_home": context_home,
        "context_dir": context_home,
        "index_path": context_home / str(context_config.get("index_file", "index.json")),
        "sessions_dir": sessions_dir,
        "legacy_path": context_home / str(context_config.get("file_name", "voice_context.json")),
    }


def normalize_term_items(value: Any) -> list[str]:
    if value is None or value == "":
        return []
    if isinstance(value, str):
        raw_items = value.splitlines() if "\n" in value else value.split(",")
    elif isinstance(value, (list, tuple, set)):
        raw_items = list(value)
    else:
        raw_items = [value]

    normalized: list[str] = []
    for item in raw_items:
        cleaned = sanitize_text(item, max_chars=120).strip()
        if cleaned and cleaned not in normalized:
            normalized.append(cleaned)
        if len(normalized) >= TERM_BANK_LIMIT:
            break
    return normalized


def normalize_term_bank(raw: dict[str, Any] | None) -> dict[str, Any]:
    payload = raw or {}
    if isinstance(payload, dict) and isinstance(payload.get("term_bank"), dict):
        payload = payload.get("term_bank", {})

    term_bank = empty_term_bank()
    if not isinstance(payload, dict):
        return term_bank

    for field in TERM_BANK_FIELDS:
        term_bank[field] = normalize_term_items(payload.get(field))
    return term_bank


def merge_term_bank(existing: dict[str, Any] | None, updates: dict[str, Any] | None) -> dict[str, Any]:
    merged = normalize_term_bank(existing)
    payload = updates or {}
    if not isinstance(payload, dict):
        return merged

    for field in TERM_BANK_FIELDS:
        if field not in payload or payload.get(field) is None:
            continue
        fresh_items = normalize_term_items(payload.get(field))
        combined: list[str] = []
        for item in fresh_items + merged[field]:
            if item and item not in combined:
                combined.append(item)
            if len(combined) >= TERM_BANK_LIMIT:
                break
        merged[field] = combined
    return merged


def _profile_session_path(paths: dict[str, Path], context_id: str) -> Path:
    return paths["sessions_dir"] / f"{context_id}.json"


def default_profile_entry(*, file_path: str = "") -> dict[str, Any]:
    return {
        "profile_id": DEFAULT_PROFILE_ID,
        "app": "default",
        "session_id": "default",
        "project_name": "default-coding",
        "project_root": "",
        "context_id": DEFAULT_PROFILE_ID,
        "last_active": "",
        "file_path": file_path,
    }


def _normalize_profile_entry(item: dict[str, Any], paths: dict[str, Path]) -> dict[str, Any]:
    profile_id = sanitize_text(item.get("profile_id", ""), max_chars=120)
    context_id = sanitize_text(item.get("context_id", ""), max_chars=120)
    return {
        "profile_id": profile_id,
        "app": sanitize_text(item.get("app", ""), max_chars=80),
        "session_id": sanitize_text(item.get("session_id", ""), max_chars=120),
        "project_name": sanitize_text(item.get("project_name", ""), max_chars=200),
        "project_root": sanitize_text(item.get("project_root", ""), max_chars=500),
        "context_id": context_id,
        "last_active": sanitize_text(item.get("last_active", ""), max_chars=80),
        "file_path": sanitize_text(
            item.get("file_path", str(_profile_session_path(paths, context_id)) if context_id else ""),
            max_chars=500,
        ),
    }


def _normalize_index_payload(raw_index: dict[str, Any] | None, paths: dict[str, Path]) -> dict[str, Any]:
    index = empty_index()
    payload = raw_index if isinstance(raw_index, dict) else {}
    index["active_context_id"] = sanitize_text(payload.get("active_context_id", ""), max_chars=120)
    index["active_session_id"] = sanitize_text(payload.get("active_session_id", ""), max_chars=120)
    index["active_profile_id"] = sanitize_text(payload.get("active_profile_id", ""), max_chars=120)

    contexts = payload.get("contexts", [])
    if isinstance(contexts, list):
        normalized_contexts = []
        for item in contexts:
            if not isinstance(item, dict):
                continue
            normalized_contexts.append(
                {
                    "context_id": sanitize_text(item.get("context_id", ""), max_chars=120),
                    "session_id": sanitize_text(item.get("session_id", ""), max_chars=120),
                    "project_name": sanitize_text(item.get("project_name", ""), max_chars=200),
                    "project_root": sanitize_text(item.get("project_root", ""), max_chars=500),
                    "agent": sanitize_text(item.get("agent", ""), max_chars=120),
                    "app": sanitize_text(item.get("app", ""), max_chars=80),
                    "profile_id": sanitize_text(item.get("profile_id", ""), max_chars=120),
                    "last_active": sanitize_text(item.get("last_active", ""), max_chars=80),
                    "file_path": sanitize_text(item.get("file_path", ""), max_chars=500),
                }
            )
        index["contexts"] = normalized_contexts

    profiles = payload.get("profiles", [])
    normalized_profiles = []
    if isinstance(profiles, list):
        for item in profiles:
            if not isinstance(item, dict):
                continue
            normalized_profiles.append(_normalize_profile_entry(item, paths))

    if not any(item.get("profile_id") == DEFAULT_PROFILE_ID for item in normalized_profiles):
        normalized_profiles.append(
            default_profile_entry(file_path=str(_profile_session_path(paths, DEFAULT_PROFILE_ID)))
        )
    normalized_profiles.sort(key=lambda item: item.get("last_active", ""), reverse=True)
    index["profiles"] = normalized_profiles
    return index


def _find_profile(index: dict[str, Any], profile_id: str) -> dict[str, Any] | None:
    for item in index.get("profiles", []):
        if item.get("profile_id") == profile_id:
            return item
    return None


def resolve_context_scope(
    *,
    config: dict[str, Any] | None = None,
    profile_id: str | None = None,
    session_id: str | None = None,
    context_id: str | None = None,
    project_root: str | None = None,
    app: str | None = None,
    agent: str | None = None,
    default_app: str | None = None,
    cwd: str | Path | None = None,
) -> dict[str, Any]:
    loaded_config = config or load_config()
    paths = _context_paths(loaded_config)
    index = _normalize_index_payload(load_json_file(paths["index_path"]), paths)

    requested_profile_id = (profile_id or "").strip()
    env_session_id = os.environ.get("VOICE_CONTEXT_SESSION_ID", "").strip()
    env_context_id = os.environ.get("VOICE_CONTEXT_ID", "").strip()
    env_project_root = os.environ.get("VOICE_CONTEXT_PROJECT_ROOT", "").strip()
    env_agent = os.environ.get("VOICE_CONTEXT_AGENT", "").strip()
    requested_session_id = (session_id or "").strip() or env_session_id
    requested_context_id = (context_id or "").strip() or env_context_id

    profile_entry: dict[str, Any] | None = None
    uses_active_profile = False
    if not requested_session_id and not requested_context_id:
        selected_profile_id = requested_profile_id or index.get("active_profile_id", "") or DEFAULT_PROFILE_ID
        profile_entry = _find_profile(index, selected_profile_id)
        if profile_entry is None and selected_profile_id == DEFAULT_PROFILE_ID:
            profile_entry = default_profile_entry(file_path=str(_profile_session_path(paths, DEFAULT_PROFILE_ID)))
        uses_active_profile = not requested_profile_id and bool(index.get("active_profile_id", ""))

    requested_project_root = (project_root or "").strip()
    if not requested_project_root and profile_entry is not None:
        requested_project_root = str(profile_entry.get("project_root", "")).strip()
    if not requested_project_root and profile_entry is None:
        requested_project_root = env_project_root or str(cwd or os.getcwd())
    normalized_project_root = (
        normalize_project_root_value(requested_project_root)
        if requested_project_root
        else ""
    )
    fallback_context_id = build_fallback_context_id(normalized_project_root or DEFAULT_PROFILE_ID)

    resolved_context_id = requested_session_id or requested_context_id
    if not resolved_context_id and profile_entry is not None:
        resolved_context_id = str(profile_entry.get("context_id", "")).strip()
    resolved_context_id = resolved_context_id or fallback_context_id

    resolved_session_id = requested_session_id
    if not resolved_session_id and profile_entry is not None:
        resolved_session_id = str(profile_entry.get("session_id", "")).strip()
    resolved_session_id = resolved_session_id or resolved_context_id

    resolved_profile_id = requested_profile_id
    if not resolved_profile_id:
        if profile_entry is not None:
            resolved_profile_id = str(profile_entry.get("profile_id", "")).strip()
        elif requested_session_id:
            resolved_profile_id = requested_session_id
        elif requested_context_id:
            resolved_profile_id = requested_context_id
        else:
            resolved_profile_id = DEFAULT_PROFILE_ID

    resolved_app = (app or "").strip()
    if not resolved_app and profile_entry is not None:
        resolved_app = str(profile_entry.get("app", "")).strip()
    resolved_app = resolved_app or (default_app or "").strip() or "cli"
    resolved_agent = (agent or "").strip() or env_agent or "codex"
    project_name = (
        sanitize_text(profile_entry.get("project_name", ""), max_chars=200)
        if profile_entry is not None and profile_entry.get("project_name")
        else (Path(normalized_project_root).name if normalized_project_root else "default-coding")
    ) or "default-coding"

    return {
        "context_home": paths["context_home"],
        "context_dir": paths["context_dir"],
        "index_path": paths["index_path"],
        "sessions_dir": paths["sessions_dir"],
        "session_path": _profile_session_path(paths, resolved_context_id),
        "legacy_path": paths["legacy_path"],
        "profile_id": resolved_profile_id,
        "context_id": resolved_context_id,
        "session_id": resolved_session_id,
        "requested_profile_id": requested_profile_id,
        "requested_session_id": requested_session_id,
        "requested_context_id": requested_context_id,
        "uses_legacy_fallback": not requested_profile_id and not requested_session_id and not requested_context_id and not uses_active_profile,
        "project_root": normalized_project_root,
        "project_name": project_name,
        "app": resolved_app,
        "agent": resolved_agent,
        "index": index,
    }


def get_context_path(
    config: dict[str, Any] | None = None,
    *,
    scope: dict[str, Any] | None = None,
    profile_id: str | None = None,
    session_id: str | None = None,
    context_id: str | None = None,
    project_root: str | None = None,
    app: str | None = None,
    agent: str | None = None,
) -> Path:
    resolved_scope = scope or resolve_context_scope(
        config=config,
        profile_id=profile_id,
        session_id=session_id,
        context_id=context_id,
        project_root=project_root,
        app=app,
        agent=agent,
    )
    return resolved_scope["session_path"]


def get_context_status(
    config: dict[str, Any] | None = None,
    *,
    scope: dict[str, Any] | None = None,
    profile_id: str | None = None,
    session_id: str | None = None,
    context_id: str | None = None,
    project_root: str | None = None,
    app: str | None = None,
    agent: str | None = None,
) -> dict[str, Any]:
    resolved_scope = scope or resolve_context_scope(
        config=config,
        profile_id=profile_id,
        session_id=session_id,
        context_id=context_id,
        project_root=project_root,
        app=app,
        agent=agent,
    )
    full_context = load_full_context(config=config, scope=resolved_scope)
    return {
        "context_home": str(resolved_scope["context_home"]),
        "index_path": str(resolved_scope["index_path"]),
        "context_path": str(resolved_scope["session_path"]),
        "profile_id": resolved_scope["profile_id"],
        "context_id": resolved_scope["context_id"],
        "session_id": resolved_scope["session_id"],
        "app": resolved_scope["app"],
        "project_root": resolved_scope["project_root"],
        "summary_dirty": bool(full_context.get("summary_dirty", False)),
        "turns_since_summary": _safe_int(full_context.get("turns_since_summary", 0)),
        "recent_messages_count": len(full_context.get("recent_messages", [])),
    }


def normalize_list(value: Any) -> list[str]:
    if value is None or value == "":
        return []
    if isinstance(value, str):
        if "\n" in value:
            raw_items = value.splitlines()
        else:
            raw_items = [item.strip() for item in value.split(",")]
    elif isinstance(value, (list, tuple, set)):
        raw_items = list(value)
    else:
        raw_items = [value]
    return [
        str(sanitize_value(item, max_chars=300)).strip()
        for item in raw_items
        if item is not None and str(item).strip()
    ][:20]


def normalize_context(raw: dict[str, Any] | None) -> dict[str, Any]:
    payload = raw or {}
    if isinstance(payload, dict) and isinstance(payload.get("summary"), dict):
        payload = payload["summary"]

    context = empty_context()
    if not isinstance(payload, dict):
        return context

    for field in CONTEXT_FIELDS:
        value = payload.get(field, context[field])
        if field in LIST_FIELDS:
            context[field] = normalize_list(value)
        else:
            context[field] = sanitize_value(value)
    return context


def empty_session_context(scope: dict[str, Any]) -> dict[str, Any]:
    return {
        "profile_id": scope["profile_id"],
        "context_id": scope["context_id"],
        "session_id": scope["session_id"],
        "app": scope["app"],
        "project_name": scope["project_name"],
        "project_root": scope["project_root"],
        "agent": scope["agent"],
        "summary": empty_context(),
        "term_bank": empty_term_bank(),
        "recent_messages": [],
        "summary_dirty": False,
        "turns_since_summary": 0,
        "updated_at": "",
    }


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return max(int(value), 0)
    except (TypeError, ValueError):
        return default


def sanitize_message_text(text: Any, max_chars: int) -> str:
    cleaned = sanitize_text(text, max_chars=max_chars * 2)
    cleaned = MESSAGE_CODE_BLOCK_PATTERN.sub("[CODE BLOCK REDACTED]", cleaned)
    if cleaned.count("\n") >= 12 and len(cleaned) >= 400 and LIKELY_CODE_PATTERN.search(cleaned):
        first_line = cleaned.splitlines()[0][:160].strip()
        return sanitize_text(f"{first_line} [LONG CODE BLOCK REDACTED]", max_chars=220)
    if len(cleaned) > max_chars:
        cleaned = sanitize_text(cleaned, max_chars=max_chars)
    return cleaned


def normalize_recent_message(
    raw: dict[str, Any] | None,
    *,
    scope: dict[str, Any],
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    loaded_config = config or load_config()
    context_config = loaded_config.get("context", {})
    max_chars = int(context_config.get("max_message_chars", 1200))
    payload = raw or {}

    return {
        "role": sanitize_text(payload.get("role", ""), max_chars=40),
        "content": sanitize_message_text(payload.get("content", ""), max_chars=max_chars),
        "summary": sanitize_message_text(payload.get("summary", ""), max_chars=max_chars),
        "session_id": sanitize_text(payload.get("session_id", scope["session_id"]), max_chars=120),
        "time": sanitize_text(payload.get("time", utc_now_iso()), max_chars=80),
    }


def normalize_full_context(
    raw: dict[str, Any] | None,
    *,
    scope: dict[str, Any],
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    loaded_config = config or load_config()
    context_config = loaded_config.get("context", {})
    max_messages = int(context_config.get("recent_messages_max", 8))
    payload = raw or {}
    context = empty_session_context(scope)

    if isinstance(payload, dict):
        context["profile_id"] = sanitize_text(payload.get("profile_id", scope["profile_id"]), max_chars=120)
        context["context_id"] = sanitize_text(payload.get("context_id", scope["context_id"]), max_chars=120)
        context["session_id"] = sanitize_text(payload.get("session_id", scope["session_id"]), max_chars=120)
        context["app"] = sanitize_text(payload.get("app", scope["app"]), max_chars=80)
        context["project_name"] = sanitize_text(payload.get("project_name", scope["project_name"]), max_chars=200)
        context["project_root"] = sanitize_text(payload.get("project_root", scope["project_root"]), max_chars=500)
        context["agent"] = sanitize_text(payload.get("agent", scope["agent"]), max_chars=120)
        context["summary"] = normalize_context(payload)
        context["term_bank"] = normalize_term_bank(payload)
        context["recent_messages"] = [
            normalize_recent_message(item, scope=scope, config=loaded_config)
            for item in payload.get("recent_messages", [])[-max_messages:]
            if isinstance(item, dict)
        ]
        context["summary_dirty"] = bool(payload.get("summary_dirty", False))
        context["turns_since_summary"] = _safe_int(payload.get("turns_since_summary", 0))
        context["updated_at"] = sanitize_text(payload.get("updated_at", context["summary"]["updated_at"]), max_chars=80)

    if not context["updated_at"]:
        context["updated_at"] = context["summary"]["updated_at"]
    return context


def load_json_file(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        with path.open("r", encoding="utf-8") as file:
            data = json.load(file)
    except Exception:
        return None
    return data if isinstance(data, dict) else None


def load_index(config: dict[str, Any] | None = None, *, scope: dict[str, Any] | None = None) -> dict[str, Any]:
    resolved_scope = scope or resolve_context_scope(config=config)
    return _normalize_index_payload(load_json_file(resolved_scope["index_path"]), _context_paths(config))


def save_index(index: dict[str, Any], config: dict[str, Any] | None = None, *, scope: dict[str, Any] | None = None) -> dict[str, Any]:
    resolved_scope = scope or resolve_context_scope(config=config)
    resolved_scope["context_dir"].mkdir(parents=True, exist_ok=True)
    normalized = _normalize_index_payload(index, _context_paths(config))
    with resolved_scope["index_path"].open("w", encoding="utf-8") as file:
        file.write(json_dumps(normalized) + "\n")
    return normalized


def build_session_from_legacy(scope: dict[str, Any], config: dict[str, Any] | None = None) -> dict[str, Any]:
    legacy_summary = normalize_context(load_json_file(scope["legacy_path"]))
    full_context = empty_session_context(scope)
    full_context["summary"] = legacy_summary
    full_context["updated_at"] = legacy_summary["updated_at"]
    return full_context


def load_full_context(
    config: dict[str, Any] | None = None,
    *,
    scope: dict[str, Any] | None = None,
    profile_id: str | None = None,
    session_id: str | None = None,
    context_id: str | None = None,
    project_root: str | None = None,
    app: str | None = None,
    agent: str | None = None,
) -> dict[str, Any]:
    resolved_scope = scope or resolve_context_scope(
        config=config,
        profile_id=profile_id,
        session_id=session_id,
        context_id=context_id,
        project_root=project_root,
        app=app,
        agent=agent,
    )
    raw_context = load_json_file(resolved_scope["session_path"])
    if isinstance(raw_context, dict):
        return normalize_full_context(raw_context, scope=resolved_scope, config=config)

    if resolved_scope["uses_legacy_fallback"] and resolved_scope["legacy_path"].exists():
        return build_session_from_legacy(resolved_scope, config=config)

    return empty_session_context(resolved_scope)


def save_full_context(
    context: dict[str, Any],
    config: dict[str, Any] | None = None,
    *,
    scope: dict[str, Any] | None = None,
) -> dict[str, Any]:
    resolved_scope = scope or resolve_context_scope(config=config)
    normalized = normalize_full_context(context, scope=resolved_scope, config=config)
    resolved_scope["sessions_dir"].mkdir(parents=True, exist_ok=True)
    with resolved_scope["session_path"].open("w", encoding="utf-8") as file:
        file.write(json_dumps(normalized) + "\n")
    update_context_index(normalized, config=config, scope=resolved_scope)
    return normalized


def list_profiles(config: dict[str, Any] | None = None, *, scope: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    resolved_scope = scope or resolve_context_scope(config=config)
    index = load_index(config=config, scope=resolved_scope)
    profiles = list(index.get("profiles", []))
    profiles.sort(key=lambda item: item.get("last_active", ""), reverse=True)
    return profiles


def get_profile(
    profile_id: str,
    config: dict[str, Any] | None = None,
    *,
    scope: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    resolved_scope = scope or resolve_context_scope(config=config)
    return _find_profile(load_index(config=config, scope=resolved_scope), profile_id)


def get_current_profile(config: dict[str, Any] | None = None, *, scope: dict[str, Any] | None = None) -> dict[str, Any]:
    resolved_scope = scope or resolve_context_scope(config=config)
    profile = get_profile(resolved_scope["profile_id"], config=config, scope=resolved_scope)
    if profile is not None:
        return profile
    return default_profile_entry(file_path=str(resolved_scope["session_path"]))


def create_or_update_profile(
    *,
    profile_id: str,
    app: str,
    session_id: str,
    project_root: str,
    context_id: str = "",
    project_name: str = "",
    last_active: str = "",
    config: dict[str, Any] | None = None,
    scope: dict[str, Any] | None = None,
    set_active: bool = False,
) -> dict[str, Any]:
    loaded_config = config or load_config()
    paths = _context_paths(loaded_config)
    normalized_project_root = normalize_project_root_value(project_root) if project_root else ""
    resolved_context_id = sanitize_text(context_id or session_id or build_fallback_context_id(normalized_project_root or DEFAULT_PROFILE_ID), max_chars=120)
    entry = _normalize_profile_entry(
        {
            "profile_id": profile_id,
            "app": app,
            "session_id": session_id or resolved_context_id,
            "project_name": project_name or (Path(normalized_project_root).name if normalized_project_root else profile_id),
            "project_root": normalized_project_root,
            "context_id": resolved_context_id,
            "last_active": last_active,
            "file_path": str(_profile_session_path(paths, resolved_context_id)),
        },
        paths,
    )
    resolved_scope = scope or resolve_context_scope(
        config=loaded_config,
        profile_id=profile_id,
        session_id=entry["session_id"],
        context_id=entry["context_id"],
        project_root=entry["project_root"],
        app=entry["app"],
        default_app=entry["app"],
    )
    index = load_index(config=loaded_config, scope=resolved_scope)
    profiles = [item for item in index.get("profiles", []) if item.get("profile_id") != profile_id]
    profiles.append(entry)
    profiles.sort(key=lambda item: item.get("last_active", ""), reverse=True)
    index["profiles"] = profiles
    if set_active:
        index["active_profile_id"] = profile_id
    save_index(index, config=loaded_config, scope=resolved_scope)
    return entry


def set_active_profile(
    profile_id: str,
    config: dict[str, Any] | None = None,
    *,
    scope: dict[str, Any] | None = None,
) -> dict[str, Any]:
    resolved_scope = scope or resolve_context_scope(config=config)
    index = load_index(config=config, scope=resolved_scope)
    profile = _find_profile(index, profile_id)
    if profile is None:
        raise ValueError(f"未找到 profile: {profile_id}")
    index["active_profile_id"] = profile_id
    save_index(index, config=config, scope=resolved_scope)
    return profile


def update_context_index(
    full_context: dict[str, Any],
    *,
    config: dict[str, Any] | None = None,
    scope: dict[str, Any] | None = None,
) -> dict[str, Any]:
    resolved_scope = scope or resolve_context_scope(config=config)
    index = load_index(config=config, scope=resolved_scope)
    entry = {
        "profile_id": full_context.get("profile_id", resolved_scope["profile_id"]),
        "context_id": full_context["context_id"],
        "session_id": full_context["session_id"],
        "app": full_context.get("app", resolved_scope["app"]),
        "project_name": full_context["project_name"],
        "project_root": full_context["project_root"],
        "agent": full_context["agent"],
        "last_active": full_context["updated_at"],
        "file_path": str(resolved_scope["session_path"]),
    }

    contexts = [item for item in index["contexts"] if item.get("context_id") != full_context["context_id"]]
    contexts.append(entry)
    contexts.sort(key=lambda item: item.get("last_active", ""), reverse=True)
    index["contexts"] = contexts
    index["active_context_id"] = full_context["context_id"]
    index["active_session_id"] = full_context["session_id"]
    index["active_profile_id"] = full_context.get("profile_id", resolved_scope["profile_id"])
    profile_entry = _normalize_profile_entry(
        {
            "profile_id": full_context.get("profile_id", resolved_scope["profile_id"]),
            "app": full_context.get("app", resolved_scope["app"]),
            "session_id": full_context["session_id"],
            "project_name": full_context["project_name"],
            "project_root": full_context["project_root"],
            "context_id": full_context["context_id"],
            "last_active": full_context["updated_at"],
            "file_path": str(resolved_scope["session_path"]),
        },
        _context_paths(config),
    )
    profiles = [item for item in index.get("profiles", []) if item.get("profile_id") != profile_entry["profile_id"]]
    profiles.append(profile_entry)
    profiles.sort(key=lambda item: item.get("last_active", ""), reverse=True)
    index["profiles"] = profiles
    return save_index(index, config=config, scope=resolved_scope)


def load_context(
    config: dict[str, Any] | None = None,
    *,
    scope: dict[str, Any] | None = None,
    profile_id: str | None = None,
    session_id: str | None = None,
    context_id: str | None = None,
    project_root: str | None = None,
    app: str | None = None,
    agent: str | None = None,
) -> dict[str, Any]:
    return load_full_context(
        config=config,
        scope=scope,
        profile_id=profile_id,
        session_id=session_id,
        context_id=context_id,
        project_root=project_root,
        app=app,
        agent=agent,
    )["summary"]


def clear_context(
    config: dict[str, Any] | None = None,
    *,
    scope: dict[str, Any] | None = None,
    profile_id: str | None = None,
    session_id: str | None = None,
    context_id: str | None = None,
    project_root: str | None = None,
    app: str | None = None,
    agent: str | None = None,
) -> dict[str, Any]:
    resolved_scope = scope or resolve_context_scope(
        config=config,
        profile_id=profile_id,
        session_id=session_id,
        context_id=context_id,
        project_root=project_root,
        app=app,
        agent=agent,
    )
    full_context = empty_session_context(resolved_scope)
    now = utc_now_iso()
    full_context["summary"]["updated_at"] = now
    full_context["updated_at"] = now
    return save_full_context(full_context, config=config, scope=resolved_scope)


def update_voice_context(
    *,
    current_goal: str | None = None,
    latest_dynamic: str | None = None,
    recent_ai_summary: str | None = None,
    recent_user_intent: str | None = None,
    important_terms: list[str] | str | None = None,
    possible_files: list[str] | str | None = None,
    constraints: list[str] | str | None = None,
    next_likely_commands: list[str] | str | None = None,
    term_bank: dict[str, Any] | None = None,
    config: dict[str, Any] | None = None,
    scope: dict[str, Any] | None = None,
    profile_id: str | None = None,
    session_id: str | None = None,
    context_id: str | None = None,
    project_root: str | None = None,
    app: str | None = None,
    agent: str | None = None,
) -> dict[str, Any]:
    resolved_scope = scope or resolve_context_scope(
        config=config,
        profile_id=profile_id,
        session_id=session_id,
        context_id=context_id,
        project_root=project_root,
        app=app,
        agent=agent,
    )
    full_context = load_full_context(config=config, scope=resolved_scope)
    summary = normalize_context(full_context)
    updates = {
        "current_goal": current_goal,
        "latest_dynamic": latest_dynamic,
        "recent_ai_summary": recent_ai_summary,
        "recent_user_intent": recent_user_intent,
        "important_terms": important_terms,
        "possible_files": possible_files,
        "constraints": constraints,
        "next_likely_commands": next_likely_commands,
    }

    for field, value in updates.items():
        if value is None:
            continue
        if field in LIST_FIELDS:
            summary[field] = normalize_list(value)
        else:
            summary[field] = sanitize_value(value)

    now = utc_now_iso()
    summary["updated_at"] = now
    merged_term_bank = merge_term_bank(full_context.get("term_bank"), term_bank)
    full_context["profile_id"] = resolved_scope["profile_id"]
    full_context["context_id"] = resolved_scope["context_id"]
    full_context["session_id"] = resolved_scope["session_id"]
    full_context["app"] = resolved_scope["app"]
    full_context["project_name"] = resolved_scope["project_name"]
    full_context["project_root"] = resolved_scope["project_root"]
    full_context["agent"] = resolved_scope["agent"]
    full_context["summary"] = summary
    full_context["term_bank"] = merged_term_bank
    full_context["summary_dirty"] = False
    full_context["turns_since_summary"] = 0
    full_context["updated_at"] = now
    return save_full_context(full_context, config=config, scope=resolved_scope)


def update_voice_context_json(
    payload_json: str,
    *,
    config: dict[str, Any] | None = None,
    scope: dict[str, Any] | None = None,
    profile_id: str | None = None,
    session_id: str | None = None,
    context_id: str | None = None,
    project_root: str | None = None,
    app: str | None = None,
    agent: str | None = None,
) -> dict[str, Any]:
    base_scope = scope or resolve_context_scope(
        config=config,
        profile_id=profile_id,
        session_id=session_id,
        context_id=context_id,
        project_root=project_root,
        app=app,
        agent=agent,
    )

    if payload_json is None or not str(payload_json).strip():
        return {
            "ok": False,
            "error": "payload_json 不能为空。",
            "used_context_id": base_scope["context_id"],
            "used_session_id": base_scope["session_id"],
            "context_path": str(base_scope["session_path"]),
            "parsed_payload": {},
            "applied_updates": {},
            "context": load_full_context(config=config, scope=base_scope),
        }

    def parse_json_object(value: Any, label: str) -> tuple[dict[str, Any] | None, str | None]:
        if isinstance(value, dict):
            data = value
        elif isinstance(value, str):
            if not value.strip():
                return None, f"{label} 不能为空。"
            try:
                data = json.loads(value)
            except json.JSONDecodeError as exc:
                return None, f"{label} 不是合法 JSON：{exc}"
        else:
            return None, f"{label} 必须是 JSON object 或 JSON 字符串。"

        if not isinstance(data, dict):
            return None, f"{label} 必须解析成 JSON object。"
        return data, None

    outer_payload, parse_error = parse_json_object(payload_json, "payload_json")
    if parse_error:
        return {
            "ok": False,
            "error": parse_error,
            "used_context_id": base_scope["context_id"],
            "used_session_id": base_scope["session_id"],
            "context_path": str(base_scope["session_path"]),
            "parsed_payload": {},
            "applied_updates": {},
            "context": load_full_context(config=config, scope=base_scope),
        }

    envelope_scope: dict[str, str] = {}
    parsed_payload = outer_payload
    warning = ""

    if any(key in outer_payload for key in JSON_ENVELOPE_KEYS):
        for field in ("profile_id", "session_id", "context_id", "project_root", "app", "agent"):
            value = outer_payload.get(field)
            if value is not None and str(value).strip():
                envelope_scope[field] = str(value).strip()

        inner_payload = outer_payload.get("payload_json")
        parsed_payload, parse_error = parse_json_object(inner_payload, "payload_json.payload_json")
        if parse_error:
            provisional_scope = resolve_context_scope(
                config=config,
                profile_id=(profile_id or "").strip() or envelope_scope.get("profile_id") or base_scope["profile_id"],
                session_id=(session_id or "").strip() or envelope_scope.get("session_id") or base_scope["session_id"],
                context_id=(context_id or "").strip() or envelope_scope.get("context_id") or base_scope["context_id"],
                project_root=(project_root or "").strip() or envelope_scope.get("project_root") or base_scope["project_root"],
                app=(app or "").strip() or envelope_scope.get("app", "") or base_scope["app"],
                agent=(agent or "").strip() or envelope_scope.get("agent") or base_scope["agent"],
            )
            return {
                "ok": False,
                "error": parse_error,
                "used_context_id": provisional_scope["context_id"],
                "used_session_id": provisional_scope["session_id"],
                "context_path": str(provisional_scope["session_path"]),
                "parsed_payload": sanitize_value(outer_payload, max_chars=1200),
                "applied_updates": {},
                "context": load_full_context(config=config, scope=provisional_scope),
            }

    applied_updates = {
        field: parsed_payload.get(field)
        for field in SUMMARY_UPDATE_FIELDS
        if field in parsed_payload and parsed_payload.get(field) is not None
    }
    if "term_bank" in parsed_payload and parsed_payload.get("term_bank") is not None:
        applied_updates["term_bank"] = parsed_payload.get("term_bank")

    def pick_scope_value(field: str, explicit_value: str | None) -> str:
        explicit_text = (explicit_value or "").strip()
        if explicit_text:
            return explicit_text
        envelope_text = envelope_scope.get(field, "").strip()
        if envelope_text:
            return envelope_text
        return str(base_scope.get(field, "")).strip()

    resolved_scope = resolve_context_scope(
        config=config,
        profile_id=pick_scope_value("profile_id", profile_id) or None,
        session_id=pick_scope_value("session_id", session_id) or None,
        context_id=pick_scope_value("context_id", context_id) or None,
        project_root=pick_scope_value("project_root", project_root) or None,
        app=(app or "").strip() or envelope_scope.get("app", "") or base_scope["app"],
        agent=pick_scope_value("agent", agent) or None,
    )

    if not applied_updates:
        return {
            "ok": False,
            "warning": "JSON 解析成功，但没有可更新字段。",
            "used_context_id": resolved_scope["context_id"],
            "used_session_id": resolved_scope["session_id"],
            "context_path": str(resolved_scope["session_path"]),
            "parsed_payload": sanitize_value(parsed_payload, max_chars=1200),
            "applied_updates": {},
            "context": load_full_context(config=config, scope=resolved_scope),
        }

    normalized_applied_updates = {
        field: (
            normalize_term_bank(value)
            if field == "term_bank"
            else normalize_list(value) if field in LIST_FIELDS else sanitize_value(value)
        )
        for field, value in applied_updates.items()
    }

    result = update_voice_context(
        config=config,
        scope=resolved_scope,
        **applied_updates,
    )

    if envelope_scope:
        warning = "检测到 Inspector envelope JSON，已自动解包并应用外层 scope。"

    return {
        "ok": True,
        "warning": warning,
        "used_context_id": resolved_scope["context_id"],
        "used_session_id": resolved_scope["session_id"],
        "context_path": str(resolved_scope["session_path"]),
        "parsed_payload": sanitize_value(parsed_payload, max_chars=1200),
        "applied_updates": normalized_applied_updates,
        "context": result,
    }


def append_chat_event(
    *,
    role: str,
    content: str,
    summary: str | None = None,
    config: dict[str, Any] | None = None,
    scope: dict[str, Any] | None = None,
    profile_id: str | None = None,
    session_id: str | None = None,
    context_id: str | None = None,
    project_root: str | None = None,
    app: str | None = None,
    agent: str | None = None,
) -> dict[str, Any]:
    resolved_scope = scope or resolve_context_scope(
        config=config,
        profile_id=profile_id,
        session_id=session_id,
        context_id=context_id,
        project_root=project_root,
        app=app,
        agent=agent,
    )
    loaded_config = config or load_config()
    max_messages = int(loaded_config.get("context", {}).get("recent_messages_max", 8))
    full_context = load_full_context(config=loaded_config, scope=resolved_scope)
    event = normalize_recent_message(
        {
            "role": role,
            "content": content,
            "summary": summary or "",
            "session_id": resolved_scope["session_id"],
            "time": utc_now_iso(),
        },
        scope=resolved_scope,
        config=loaded_config,
    )
    full_context["recent_messages"] = (full_context.get("recent_messages", []) + [event])[-max_messages:]
    full_context["summary_dirty"] = True
    full_context["turns_since_summary"] = _safe_int(full_context.get("turns_since_summary", 0)) + 1
    full_context["updated_at"] = utc_now_iso()
    return save_full_context(full_context, config=loaded_config, scope=resolved_scope)


def _short_term_bank_lines(term_bank: dict[str, Any], *, max_categories: int = 4, max_items_each: int = 4) -> list[str]:
    lines: list[str] = []
    normalized = normalize_term_bank(term_bank)
    labels = {
        "domain_terms": "domain_terms",
        "files": "files",
        "functions": "functions",
        "classes": "classes",
        "variables": "variables",
        "commands": "commands",
        "ui_terms": "ui_terms",
    }
    for field in TERM_BANK_FIELDS:
        values = normalized.get(field, [])[:max_items_each]
        if not values:
            continue
        lines.append(f"{labels[field]}: " + ", ".join(values))
        if len(lines) >= max_categories:
            break
    return lines


def context_for_asr(
    context: dict[str, Any] | None = None,
    config: dict[str, Any] | None = None,
    *,
    scope: dict[str, Any] | None = None,
) -> str:
    full_context = context if context is not None else load_full_context(config=config, scope=scope)
    loaded = normalize_context(full_context)
    parts = []
    if full_context.get("profile_id"):
        parts.append(f"profile: {full_context['profile_id']}")
    if full_context.get("app"):
        parts.append(f"app: {full_context['app']}")
    if full_context.get("session_id"):
        parts.append(f"session: {full_context['session_id']}")
    if full_context.get("project_name"):
        parts.append(f"project: {full_context['project_name']}")
    if loaded["current_goal"]:
        parts.append(f"当前目标：{loaded['current_goal']}")
    if loaded["latest_dynamic"]:
        parts.append(f"最新要求：{loaded['latest_dynamic']}")
    if loaded["important_terms"]:
        parts.append("术语：" + "，".join(loaded["important_terms"]))
    if loaded["possible_files"]:
        parts.append("可能文件：" + "，".join(loaded["possible_files"]))
    term_lines = _short_term_bank_lines(full_context.get("term_bank", {}), max_categories=3, max_items_each=4)
    if term_lines:
        parts.append("term_bank: " + " | ".join(term_lines))
    return "\n".join(parts)


def context_for_optimizer(
    context: dict[str, Any] | None = None,
    config: dict[str, Any] | None = None,
    *,
    scope: dict[str, Any] | None = None,
    recent_messages_limit: int | None = None,
) -> str:
    loaded_config = config or load_config()
    max_messages = int(loaded_config.get("context", {}).get("recent_messages_max", 8))
    limit = max(recent_messages_limit or max_messages, 0)
    full_context = normalize_full_context(
        context if context is not None else load_full_context(config=loaded_config, scope=scope),
        scope=scope or resolve_context_scope(config=loaded_config),
        config=loaded_config,
    )
    summary = normalize_context(full_context)
    lines = [
        f"profile_id: {full_context.get('profile_id', '')}",
        f"app: {full_context.get('app', '')}",
        f"session_id: {full_context.get('session_id', '')}",
        f"project_name: {full_context.get('project_name', '')}",
        f"current_goal: {summary['current_goal']}",
        f"latest_dynamic: {summary['latest_dynamic']}",
        f"recent_ai_summary: {summary['recent_ai_summary']}",
        f"recent_user_intent: {summary['recent_user_intent']}",
    ]
    for field in ["important_terms", "possible_files", "constraints", "next_likely_commands"]:
        if summary[field]:
            lines.append(f"{field}: " + ", ".join(summary[field]))
    for term_line in _short_term_bank_lines(full_context.get("term_bank", {}), max_categories=6, max_items_each=5):
        lines.append(term_line)

    recent_messages = full_context.get("recent_messages", [])[-limit:] if limit else []
    if recent_messages:
        lines.append("recent_messages:")
        for item in recent_messages:
            message_text = item.get("summary") or item.get("content") or ""
            if message_text:
                lines.append(f"- [{item.get('role', '')}] {message_text}")
    return "\n".join(line for line in lines if not line.endswith(": "))


def get_voice_context_data(
    *,
    mode: str = "current",
    config: dict[str, Any] | None = None,
    scope: dict[str, Any] | None = None,
    profile_id: str | None = None,
    session_id: str | None = None,
    context_id: str | None = None,
    project_root: str | None = None,
    app: str | None = None,
    agent: str | None = None,
    recent_messages_limit: int | None = None,
) -> dict[str, Any] | str:
    resolved_scope = scope or resolve_context_scope(
        config=config,
        profile_id=profile_id,
        session_id=session_id,
        context_id=context_id,
        project_root=project_root,
        app=app,
        agent=agent,
    )
    selected_mode = (mode or "current").strip().lower()
    full_context = load_full_context(config=config, scope=resolved_scope)
    summary = normalize_context(full_context)

    if selected_mode == "current":
        return summary
    if selected_mode == "asr":
        return context_for_asr(full_context, config=config, scope=resolved_scope)
    if selected_mode == "optimizer":
        return context_for_optimizer(
            full_context,
            config=config,
            scope=resolved_scope,
            recent_messages_limit=recent_messages_limit,
        )
    if selected_mode == "full":
        return full_context
    return {"error": f"不支持的 mode: {mode}"}


def format_context_for_display(
    context: dict[str, Any] | None = None,
    config: dict[str, Any] | None = None,
    *,
    mode: str = "current",
    scope: dict[str, Any] | None = None,
    recent_messages_limit: int | None = None,
) -> str:
    if context is not None:
        data: dict[str, Any] | str
        if mode == "current":
            data = normalize_context(context)
        elif mode == "full":
            data = normalize_full_context(context, scope=scope or resolve_context_scope(config=config), config=config)
        elif mode == "asr":
            return context_for_asr(context, config=config, scope=scope)
        elif mode == "optimizer":
            return context_for_optimizer(
                context,
                config=config,
                scope=scope,
                recent_messages_limit=recent_messages_limit,
            )
        else:
            data = {"error": f"不支持的 mode: {mode}"}
    else:
        data = get_voice_context_data(
            mode=mode,
            config=config,
            scope=scope,
            recent_messages_limit=recent_messages_limit,
        )

    if isinstance(data, str):
        return data
    return json_dumps(data)
