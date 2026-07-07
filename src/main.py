from __future__ import annotations

import argparse
import json
import os
import sys
import time
from copy import deepcopy
from pathlib import Path

try:
    from .context_store import (
        append_chat_event,
        clear_context,
        context_for_asr,
        create_or_update_profile,
        format_context_for_display,
        get_context_path,
        get_current_profile,
        get_context_status,
        get_profile,
        load_full_context,
        list_profiles,
        normalize_term_bank,
        resolve_context_scope,
        set_active_profile,
        update_voice_context,
    )
    from .audio_recorder import AudioDependencyError, AudioRecorder, GlobalHotkeyMonitor, HotkeyError, RecordingError
    from .prompt_editor import plan_voice_operation
    from .qwen_stt_adapter import STTError, transcribe_with_meta
    from .stt_providers import should_use_asr_context
    from .utils import PROJECT_ROOT, copy_to_clipboard, editor_provider_name, editor_status, json_dumps, load_config, replay_log_path, sanitize_text, save_replay_log_entry
    from .voice_queue import VoiceQueueManager, format_profile_compact, format_status_compact
except ImportError:
    from context_store import (
        append_chat_event,
        clear_context,
        context_for_asr,
        create_or_update_profile,
        format_context_for_display,
        get_context_path,
        get_current_profile,
        get_context_status,
        get_profile,
        load_full_context,
        list_profiles,
        normalize_term_bank,
        resolve_context_scope,
        set_active_profile,
        update_voice_context,
    )
    from audio_recorder import AudioDependencyError, AudioRecorder, GlobalHotkeyMonitor, HotkeyError, RecordingError
    from prompt_editor import plan_voice_operation
    from qwen_stt_adapter import STTError, transcribe_with_meta
    from stt_providers import should_use_asr_context
    from utils import PROJECT_ROOT, copy_to_clipboard, editor_provider_name, editor_status, json_dumps, load_config, replay_log_path, sanitize_text, save_replay_log_entry
    from voice_queue import VoiceQueueManager, format_profile_compact, format_status_compact


TERM_BANK_ARG_FIELDS = {
    "domain_terms": "domain_terms",
    "term_bank_files": "files",
    "term_bank_functions": "functions",
    "term_bank_classes": "classes",
    "term_bank_variables": "variables",
    "term_bank_commands": "commands",
    "term_bank_ui_terms": "ui_terms",
}
SPINNER_FRAMES = ["|", "/", "-", "\\"]
EDITOR_PROVIDER_CHOICES = [
    "rules",
    "ollama",
    "openai",
    "openrouter",
    "google",
    "deepseek",
    "dashscope",
    "api",
    "openai_compatible",
]
ANSI_RED = "\033[31m"
ANSI_GREEN = "\033[32m"
ANSI_DIM = "\033[2m"
ANSI_RESET = "\033[0m"
ANSI_STRIKE = "\033[9m"


def get_scope_from_args(
    args: argparse.Namespace,
    config: dict[str, object],
    *,
    default_app: str = "cli",
) -> dict[str, object]:
    return resolve_context_scope(
        config=config,
        profile_id=getattr(args, "profile", None),
        session_id=getattr(args, "session_id", None),
        context_id=getattr(args, "context_id", None),
        project_root=getattr(args, "project_root", None),
        default_app=default_app,
    )


def collect_term_bank_from_args(args: argparse.Namespace) -> dict[str, object] | None:
    payload: dict[str, object] = {}
    for arg_name, field_name in TERM_BANK_ARG_FIELDS.items():
        value = getattr(args, arg_name, None)
        if value is not None:
            payload[field_name] = value
    normalized = normalize_term_bank(payload)
    if any(normalized.get(field) for field in normalized):
        return normalized
    return None


def apply_editor_overrides(config: dict[str, object], args: argparse.Namespace) -> dict[str, object]:
    if getattr(args, "editor", None):
        config.setdefault("editor", {})
        config["editor"]["provider"] = args.editor
    if getattr(args, "timeout", None) is not None:
        config.setdefault("editor", {})
        config["editor"]["timeout_sec"] = float(args.timeout)
    if getattr(args, "model", None):
        config.setdefault("editor", {})
        current_provider = editor_provider_name(config)
        config.setdefault("editor_providers", {})
        providers = config["editor_providers"]
        if isinstance(providers, dict):
            providers.setdefault(current_provider, {})
            if isinstance(providers[current_provider], dict):
                providers[current_provider]["model"] = args.model
    return config


def get_asr_context_text(
    config: dict[str, object],
    *,
    scope: dict[str, object],
    full_context: dict[str, object] | None = None,
) -> str:
    if not should_use_asr_context(config):
        return ""
    loaded_context = full_context if full_context is not None else load_full_context(config, scope=scope)
    return context_for_asr(loaded_context, config=config, scope=scope)


def print_operation_debug(operation_result: dict[str, object]) -> None:
    print(f"operation={operation_result.get('op', '')}", file=sys.stderr)
    print(f"ignored={operation_result.get('ignored', False)}", file=sys.stderr)
    print(f"reason={operation_result.get('reason', '')}", file=sys.stderr)
    print(f"fallback={operation_result.get('fallback', False)}", file=sys.stderr)
    if operation_result.get("fallback_reason"):
        print(f"fallback_reason={operation_result.get('fallback_reason', '')}", file=sys.stderr)
    print(f"diff_items={operation_result.get('diff_items', [])}", file=sys.stderr)
    print(f"draft_blocks={operation_result.get('draft_blocks', [])}", file=sys.stderr)
    debug_info = operation_result.get("debug", {}) or {}
    for key in [
        "raw_text",
        "raw_content_preview",
        "cleaned_content_preview",
        "protected_terms",
        "qwen3_no_think_enabled",
        "no_think_injected",
        "editor_model",
        "provider_elapsed_ms",
        "raw_user_modification_request",
        "correction_raw_text",
        "correction_forced_by_key",
        "correction_triggered_by_prefix",
        "language_primary",
        "language_secondary",
        "llm_correction_prompt_preview",
        "correction_planner_raw_response",
        "correction_planner_parsed_response",
        "llm_correction_raw_response",
        "llm_correction_changed",
        "llm_correction_reason",
        "llm_position_ops",
        "normalized_internal_ops",
        "compiled_internal_ops",
        "correction_attempts",
        "final_validation_error",
        "old_blocks_preview",
        "new_blocks_preview",
        "validation_error",
        "normalization_used",
        "normalized_old",
        "normalized_new",
    ]:
        if key in debug_info:
            print(f"{key}={debug_info.get(key)}", file=sys.stderr)


def print_stt_debug(stt_result: dict[str, object]) -> None:
    print(f"audio_path={stt_result['audio_path']}", file=sys.stderr)
    print(f"asr_context_preview={stt_result['asr_context_preview']}", file=sys.stderr)
    print(f"stt_elapsed_ms={stt_result['stt_elapsed_ms']}", file=sys.stderr)
    print(f"raw_stt_preview={stt_result['raw_stt_preview']}", file=sys.stderr)
    if stt_result.get("expected_transcript_preview"):
        print(f"expected_transcript_preview={stt_result['expected_transcript_preview']}", file=sys.stderr)
    if stt_result.get("comparison_note"):
        print(f"comparison_note={stt_result['comparison_note']}", file=sys.stderr)


def should_save_replay_logs(config: dict[str, object], args: argparse.Namespace) -> bool:
    debug_config = config.get("debug", {}) if isinstance(config.get("debug", {}), dict) else {}
    configured = bool(debug_config.get("save_replay_logs", False))
    return bool(getattr(args, "save_debug_log", False) or getattr(args, "debug", False) or configured)


def normalize_eval_blocks(blocks: object) -> list[dict[str, object]]:
    if not isinstance(blocks, list):
        return []
    normalized: list[dict[str, object]] = []
    for index, item in enumerate(blocks, start=1):
        if isinstance(item, dict):
            text = sanitize_text(item.get("text", ""), max_chars=400).strip()
        else:
            text = sanitize_text(item, max_chars=400).strip()
        if not text:
            continue
        normalized.append({"id": index, "text": text})
    return normalized


def build_eval_log_entry(
    *,
    case_name: str,
    raw_stt: str,
    draft_before: list[dict[str, object]],
    expected_draft_after: list[dict[str, object]],
    operation_result: dict[str, object],
    config: dict[str, object],
    mode: str,
    exact_expected_ops_match: bool,
    expected_ops: list[dict[str, object]] | None,
    passed: bool,
    ops_match: bool | None,
) -> dict[str, object]:
    debug_payload = dict(operation_result.get("debug", {}) or {})
    editor_mode = editor_provider_name(config)
    provider_payload = debug_payload.get("provider", {})
    if not isinstance(provider_payload, dict):
        provider_payload = {}
    python_validation = {
        "accepted": not bool(operation_result.get("ignored", False)),
        "ignored": bool(operation_result.get("ignored", False)),
        "fallback": bool(operation_result.get("fallback", False)),
        "validation_error": str(debug_payload.get("validation_error", "")).strip(),
        "fallback_reason": str(operation_result.get("fallback_reason", "")).strip(),
        "normalization_used": bool(debug_payload.get("normalization_used", False)),
    }
    return {
        "case_id": case_name,
        "eval_case_name": case_name,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "generation": 1,
        "mode": mode,
        "forced_mode": mode,
        "prompt_version": str(debug_payload.get("prompt_version", "")),
        "prompt_variant": str(debug_payload.get("prompt_variant", "")),
        "raw_stt": raw_stt,
        "draft_before": draft_before,
        "system_prompt": debug_payload.get("system_prompt", ""),
        "user_prompt": debug_payload.get("user_prompt", ""),
        "llm_raw_response": debug_payload.get("llm_raw_response", ""),
        "parsed_response": debug_payload.get("parsed_response", {}),
        "parsed_ops": debug_payload.get("parsed_ops", []),
        "llm_position_ops": debug_payload.get("llm_position_ops", []),
        "normalized_internal_ops": debug_payload.get("normalized_internal_ops", []),
        "compiled_internal_ops": debug_payload.get("compiled_internal_ops", []),
        "qwen3_no_think_enabled": bool(debug_payload.get("qwen3_no_think_enabled", False)),
        "no_think_injected": bool(debug_payload.get("no_think_injected", False)),
        "editor_model": debug_payload.get("editor_model", ""),
        "provider_elapsed_ms": int(debug_payload.get("provider_elapsed_ms", 0) or 0),
        "raw_user_modification_request": debug_payload.get("raw_user_modification_request", ""),
        "correction_planner_raw_response": debug_payload.get("correction_planner_raw_response", ""),
        "correction_planner_parsed_response": debug_payload.get("correction_planner_parsed_response", {}),
        "correction_attempts": debug_payload.get("correction_attempts", []),
        "final_validation_error": debug_payload.get("final_validation_error", ""),
        "python_validation": python_validation,
        "draft_after": operation_result.get("draft_blocks", []),
        "diff_items": operation_result.get("diff_items", []),
        "reason": str(operation_result.get("reason", "")).strip(),
        "editor_elapsed_ms": int(operation_result.get("editor_elapsed_ms", 0) or 0),
        "provider": {
            "mode": provider_payload.get("mode", editor_mode),
            "model": provider_payload.get("model", ""),
            "base_url": provider_payload.get("base_url", ""),
        },
        "config_path": str(PROJECT_ROOT / "config.yaml"),
        "replay_log_path": "",
        "expected_draft_after": expected_draft_after,
        "expected_ops": expected_ops or [],
        "exact_expected_ops_match": exact_expected_ops_match,
        "ops_match": ops_match,
        "passed": passed,
    }


def copy_prompt_if_enabled(config: dict[str, object], final_prompt: str) -> None:
    output_config = config.get("output", {})
    if output_config.get("copy_to_clipboard", True):
        _, message = copy_to_clipboard(final_prompt)
        print(message, file=sys.stderr)


def compact_profile_payload(
    profile: dict[str, object],
    *,
    status: dict[str, object] | None = None,
) -> dict[str, object]:
    return {
        "profile_id": profile.get("profile_id", ""),
        "app": profile.get("app", ""),
        "session_id": profile.get("session_id", ""),
        "project_name": profile.get("project_name", ""),
        "project_root": profile.get("project_root", ""),
        "context_id": profile.get("context_id", ""),
        "context_path": (status or {}).get("context_path", ""),
        "last_active": profile.get("last_active", ""),
        "editor": profile.get("editor", {}),
    }


def print_profile_or_json(
    profile: dict[str, object],
    *,
    active_profile_id: str,
    args: argparse.Namespace,
    status: dict[str, object] | None = None,
) -> None:
    payload = compact_profile_payload(profile, status=status)
    if getattr(args, "json", False):
        print(json_dumps(payload))
        return
    effective_active_profile_id = active_profile_id or str(payload.get("profile_id", ""))
    print(
        format_profile_compact(
            payload,
            active_profile_id=effective_active_profile_id,
        )
    )
    editor_payload = payload.get("editor", {})
    if isinstance(editor_payload, dict) and editor_payload:
        provider = str(editor_payload.get("provider", "")).strip()
        print(f"  editor.provider: {provider}")
        print(f"  editor.kind: {editor_payload.get('kind', '')}")
        print(f"  editor.model: {editor_payload.get('resolved_model') or editor_payload.get('model', '')}")
        if editor_payload.get("base_url"):
            print(f"  editor.base_url: {editor_payload.get('base_url', '')}")
        if editor_payload.get("secrets_file"):
            print(f"  secrets.file: {editor_payload.get('secrets_file', '')}")
        print(f"  editor.key_loaded: {'yes' if editor_payload.get('key_loaded') else 'no'}")
        print(f"  editor.no_think: {str(bool(editor_payload.get('editor_no_think', False))).lower()}")
        print(f"  provider.no_think: {str(bool(editor_payload.get('provider_no_think', False))).lower()}")
        masked = str(editor_payload.get("key_masked", "") or "")
        if masked:
            print(f"  editor.key_masked: {masked}")
        warning = str(editor_payload.get("warning", "") or "").strip()
        if warning:
            print(f"  editor.warning: {warning}")


def print_profile_list_or_json(
    profiles: list[dict[str, object]],
    *,
    active_profile_id: str,
    args: argparse.Namespace,
) -> None:
    if getattr(args, "json", False):
        print(json_dumps(profiles))
        return
    effective_active_profile_id = active_profile_id or "default-coding"
    for index, profile in enumerate(profiles, start=1):
        print(
            format_profile_compact(
                profile,
                active_profile_id=effective_active_profile_id,
                index=index,
                include_updated=True,
            )
        )


def prompt_profile_selection(
    profiles: list[dict[str, object]],
    *,
    active_profile_id: str,
) -> dict[str, object] | None:
    if not profiles:
        print("当前没有可切换的 profile。", file=sys.stderr)
        return None
    effective_active_profile_id = active_profile_id or "default-coding"
    for index, profile in enumerate(profiles, start=1):
        print(
            format_profile_compact(
                profile,
                active_profile_id=effective_active_profile_id,
                index=index,
                include_updated=True,
            )
        )
    selection = input("输入要切换的编号（直接回车取消）：").strip()
    if not selection:
        return None
    try:
        selected_index = int(selection)
    except ValueError:
        print("请输入有效编号。", file=sys.stderr)
        return None
    if selected_index < 1 or selected_index > len(profiles):
        print("编号超出范围。", file=sys.stderr)
        return None
    return profiles[selected_index - 1]


def format_listen_header(
    *,
    profile_id: str,
    app: str,
    session_id: str,
    state: str,
) -> str:
    return f"[profile: {profile_id} | app/session: {app}/{session_id} | {state}]"


def supports_ansi_output() -> bool:
    if os.environ.get("NO_COLOR"):
        return False
    if not sys.stdout.isatty():
        return False
    if os.name == "nt":
        return True
    if os.name != "nt":
        return os.environ.get("TERM", "").lower() != "dumb"
    return False


def spinner_frame() -> str:
    return SPINNER_FRAMES[int(time.monotonic() * 8) % len(SPINNER_FRAMES)]


def format_elapsed_short(seconds_value: float) -> str:
    total_seconds = max(0, int(seconds_value))
    return f"{total_seconds // 60:02d}:{total_seconds % 60:02d}"


def build_recording_bar(recording_sec: float, max_record_sec: int | None = None) -> str:
    if not max_record_sec or max_record_sec <= 0:
        return f"Recording {format_elapsed_short(recording_sec)}"
    filled = min(10, int((recording_sec / max_record_sec) * 10))
    bar = "█" * filled + "░" * (10 - filled)
    return f"Recording {format_elapsed_short(recording_sec)} / {format_elapsed_short(max_record_sec)} {bar}"


def _style_text(text: str, *, color: str = "", strike: bool = False, ansi_enabled: bool = False) -> str:
    if not ansi_enabled:
        return text
    styled = text
    prefix = ""
    if color:
        prefix += color
    if strike:
        prefix += ANSI_STRIKE
    if not prefix:
        return styled
    return f"{prefix}{styled}{ANSI_RESET}"


def render_diff_item(
    item: dict[str, object],
    *,
    ansi_enabled: bool,
    use_color: bool,
    use_strikethrough: bool,
    fallback_diff_markers: bool,
) -> list[str]:
    kind = str(item.get("kind", ""))
    block_id = item.get("block_id", "")
    if kind == "add":
        text = f"[{block_id}] {item.get('text', '')}"
        return [("+ " + _style_text(text, color=ANSI_GREEN if use_color else "", ansi_enabled=ansi_enabled)).rstrip()]
    if kind == "delete":
        old_text = f"[{block_id}] {item.get('old', '')}"
        if ansi_enabled and use_strikethrough:
            return [("- " + _style_text(old_text, color=ANSI_RED if use_color else "", strike=True, ansi_enabled=True)).rstrip()]
        marker = "[del] " if fallback_diff_markers else "- "
        return [(marker + old_text).rstrip()]
    if kind == "replace":
        old_text = str(item.get("old", "")).strip()
        new_text = str(item.get("new", "")).strip()
        if ansi_enabled and use_color:
            old_part = _style_text(f"[{block_id}] {old_text}", color=ANSI_RED, strike=use_strikethrough, ansi_enabled=True)
            new_part = _style_text(f"[{block_id}] {new_text}", color=ANSI_GREEN, ansi_enabled=True)
            return [f"- {old_part}", f"+ {new_part}"]
        if fallback_diff_markers:
            return [f"[del] [{block_id}] {old_text}", f"[add] [{block_id}] {new_text}"]
        return [f"- [{block_id}] {old_text}", f"+ [{block_id}] {new_text}"]
    if kind == "undo":
        if item.get("operation") == "noop":
            return ["* no changes to undo"]
        return [f"* reverted change gen={item.get('generation', '')} ({item.get('operation', '')})"]
    return [sanitize_text(str(item), max_chars=200)]


def render_listen_snapshot(
    snapshot: dict[str, object],
    *,
    debug_visible: bool,
    recording_sec: float = 0.0,
    ansi_enabled: bool,
    hotkey_status: str,
    terminal_fallback: bool,
    correction_armed: bool,
    config: dict[str, object],
    extra_lines: list[str] | None = None,
) -> None:
    use_color = True
    use_strikethrough = True
    fallback_diff_markers = True
    current_segment = snapshot.get("current_segment") or {}
    generation = int(snapshot.get("generation", 0))
    copied_generation = int(snapshot.get("copied_generation", 0))
    pending_generation = int(snapshot.get("pending_generation", 0))
    state_text = str(snapshot.get("state", "idle"))
    if state_text.startswith("copied"):
        top_state = "copied"
    elif state_text.startswith("copy error"):
        top_state = "copy-error"
    elif state_text.startswith("ignored"):
        top_state = "ignored"
    elif state_text.startswith("error"):
        top_state = "error"
    elif state_text.startswith("done"):
        top_state = "done"
    else:
        top_state = state_text
    if recording_sec > 0:
        top_state = "recording"
    elif isinstance(current_segment, dict) and current_segment.get("state") == "transcribing":
        top_state = "transcribing"
    elif str(snapshot.get("opt_state", "")) == "running":
        top_state = "optimizing"

    rec_text = f"recording {format_elapsed_short(recording_sec)}" if recording_sec > 0 else "idle"
    if isinstance(current_segment, dict) and current_segment.get("state") == "transcribing":
        stt_text = f"#{current_segment.get('segment_id')} {current_segment.get('elapsed_sec', 0):.1f}s {spinner_frame()}"
    else:
        stt_text = str(snapshot.get("stt_state", "idle"))
    if str(snapshot.get("opt_state", "")) == "running":
        opt_text = f"gen={pending_generation or generation + 1} {spinner_frame()}"
    else:
        opt_text = str(snapshot.get("opt_state", "idle"))
    copy_state = str(snapshot.get("copy_state", "none"))
    copy_text = "✓" if "copied" in copy_state else copy_state

    lines: list[str] = [
        "Voice Context Local",
        f"{snapshot.get('profile_id', 'default-coding')} | {snapshot.get('app', 'default')}/{snapshot.get('session_id', 'default')} | batch #{snapshot.get('batch_id', 0)} | gen={generation} {top_state}",
        "",
        f"REC {rec_text} | STT {stt_text} | OPT {opt_text} | COPY {copy_text}",
    ]
    stt_warm_state = str(snapshot.get("stt_warm_state", "idle"))
    editor_warm_state = str(snapshot.get("editor_warm_state", "idle"))
    if stt_warm_state != "idle" or editor_warm_state != "idle":
        stt_warm_message = str(snapshot.get("stt_warm_message", "")).strip()
        editor_warm_message = str(snapshot.get("editor_warm_message", "")).strip()
        warm_line = f"Warmup: STT {stt_warm_state}"
        if stt_warm_message:
            warm_line += f" ({stt_warm_message})"
        warm_line += f" | Editor {editor_warm_state}"
        if editor_warm_message:
            warm_line += f" ({editor_warm_message})"
        lines.append(warm_line)
    if correction_armed:
        lines.append("")
        lines.append("Next recording: correction")

    lines.extend(["", str(snapshot.get("current_change_title", "Current change:"))])
    diff_items = snapshot.get("current_diff", [])
    if not diff_items:
        if str(snapshot.get("current_change_title", "")) == "Correction ignored:":
            lines.append(f'raw: "{snapshot.get("current_change_source", "")}"')
            lines.append(f'reason: {snapshot.get("current_change_reason", "") or "ignored"}')
            correction_debug = snapshot.get("current_change_debug", {}) or {}
            validation_error = str(correction_debug.get("validation_error", "")).strip()
            raw_response = str(correction_debug.get("llm_correction_raw_response", "")).strip()
            if validation_error:
                lines.append(f"validation_error: {validation_error}")
            if raw_response and validation_error:
                lines.append(f"llm_raw: {raw_response}")
        else:
            lines.append("  (no changes yet)")
    else:
        for item in diff_items:
            for line in render_diff_item(
                item,
                ansi_enabled=ansi_enabled,
                use_color=use_color,
                use_strikethrough=use_strikethrough,
                fallback_diff_markers=fallback_diff_markers,
            ):
                lines.append(line)

    lines.extend(["", "Draft:"])
    draft_blocks = snapshot.get("draft_blocks", [])
    if not draft_blocks:
        lines.append("  (empty)")
    else:
        for block in draft_blocks:
            lines.append(f"[{block.get('id', 0)}] {block.get('text', '')}")

    if debug_visible:
        lines.extend(
            [
                "",
                "Debug:",
                f"  generation={generation} | copied_generation={copied_generation} | pending_generation={pending_generation}",
                f"  editor_elapsed_ms={snapshot.get('editor_elapsed_ms', 0)} | fallback={snapshot.get('fallback', False)} | short_test_input={snapshot.get('short_test_input', False)} | skipped_project_prompt={snapshot.get('skipped_project_prompt', False)}",
                f"  current_change_source={snapshot.get('current_change_source', '')}",
                f"  last_segment={snapshot.get('segments', [])[-1]['preview'] if snapshot.get('segments', []) else ''}",
            ]
        )
        correction_debug = snapshot.get("current_change_debug", {}) or {}
        for key in [
            "raw_text",
            "raw_content_preview",
            "cleaned_content_preview",
            "protected_terms",
            "correction_raw_text",
            "correction_forced_by_key",
            "correction_triggered_by_prefix",
            "language_primary",
            "language_secondary",
            "llm_correction_prompt_preview",
            "llm_correction_raw_response",
            "llm_correction_changed",
            "llm_correction_reason",
            "old_blocks_preview",
            "new_blocks_preview",
            "validation_error",
            "normalization_used",
            "normalized_old",
            "normalized_new",
        ]:
            if key in correction_debug:
                lines.append(f"  {key}={correction_debug.get(key)}")
        fallback_reason = str(snapshot.get("fallback_reason", "")).strip()
        if fallback_reason:
            lines.append(f"  fallback_reason={fallback_reason}")
        copy_error = str(snapshot.get("copy_error", "")).strip()
        if copy_error:
            lines.append(f"  copy_error={copy_error}")

    lines.extend(
        [
            "",
            "Controls:",
            "hold=record | hold c=correction | u=undo | n=new | q=quit",
        ]
    )
    if debug_visible:
        lines.append("More: d=debug | x=clear")
        lines.append(f"Input mode: {'terminal fallback' if terminal_fallback else 'global hotkey'} | {hotkey_status}")
    elif terminal_fallback:
        lines.append("Input mode: terminal fallback")

    if extra_lines:
        lines.extend(["", *extra_lines])

    output = "\n".join(lines).rstrip() + "\n"
    if ansi_enabled:
        print("\033[2J\033[H" + output, end="", flush=True)
    else:
        print("\n" + output, end="", flush=True)


def read_console_key() -> str | None:
    if os.name != "nt":
        return None
    try:
        import msvcrt
    except ImportError:
        return None
    if not msvcrt.kbhit():
        return None
    key = msvcrt.getwch()
    if key in ("\r", "\n"):
        return "enter"
    if key == " ":
        return "space"
    return key.lower()


def command_show(args: argparse.Namespace) -> int:
    config = load_config()
    scope = get_scope_from_args(args, config, default_app="cli")
    print(
        format_context_for_display(
            config=config,
            scope=scope,
            mode=args.mode,
        )
    )
    print(f"\ncontext_path: {get_context_path(config, scope=scope)}")
    return 0


def command_status(args: argparse.Namespace) -> int:
    config = load_config()
    scope = get_scope_from_args(args, config, default_app="cli")
    status = get_context_status(config=config, scope=scope)
    if args.json:
        print(json_dumps(status))
    else:
        print(format_status_compact(status))
    return 0


def command_clear(args: argparse.Namespace) -> int:
    config = load_config()
    scope = get_scope_from_args(args, config, default_app="cli")
    clear_context(config, scope=scope)
    print(f"已清空 voice context：{get_context_path(config, scope=scope)}")
    return 0


def command_update(args: argparse.Namespace) -> int:
    config = load_config()
    scope = get_scope_from_args(args, config, default_app="cli")
    context = update_voice_context(
        current_goal=args.goal,
        latest_dynamic=args.dynamic,
        recent_ai_summary=args.ai_summary,
        recent_user_intent=args.intent,
        important_terms=args.terms,
        possible_files=args.files,
        constraints=args.constraints,
        next_likely_commands=args.commands,
        term_bank=collect_term_bank_from_args(args),
        config=config,
        scope=scope,
    )
    print(format_context_for_display(context, config=config, scope=scope, mode="current"))
    print(f"\ncontext_path: {get_context_path(config, scope=scope)}")
    return 0


def command_event(args: argparse.Namespace) -> int:
    config = load_config()
    scope = get_scope_from_args(args, config, default_app="cli")
    context = append_chat_event(
        role=args.role,
        content=args.content,
        summary=args.summary,
        config=config,
        scope=scope,
    )
    print(format_context_for_display(context, config=config, scope=scope, mode="full"))
    print(f"\ncontext_path: {get_context_path(config, scope=scope)}")
    return 0


def command_transcribe(args: argparse.Namespace) -> int:
    config = load_config()
    scope = get_scope_from_args(args, config, default_app="cli")
    asr_context = get_asr_context_text(config, scope=scope)

    try:
        stt_result = transcribe_with_meta(args.audio, asr_context)
    except STTError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    print(stt_result["text"])
    if args.debug:
        print_stt_debug(stt_result)
    return 0


def command_eval_cases(args: argparse.Namespace, *, default_mode: str = "") -> int:
    config = deepcopy(load_config())
    apply_editor_overrides(config, args)
    cases_path = Path(args.cases).expanduser()
    if not cases_path.exists():
        print(f"cases 路径不存在：{cases_path}", file=sys.stderr)
        return 1
    case_files = sorted(cases_path.glob("*.json")) if cases_path.is_dir() else [cases_path]
    if not case_files:
        print(f"没有找到 eval case：{cases_path}", file=sys.stderr)
        return 1

    eval_log_file = replay_log_path(config, kind="eval")
    total = 0
    passed = 0
    failures: list[dict[str, object]] = []

    def should_retry_eval(operation_result: dict[str, Any]) -> bool:
        debug_payload = dict(operation_result.get("debug", {}) or {})
        reason = str(operation_result.get("reason", "")).strip().lower()
        validation_error = str(debug_payload.get("validation_error", "")).strip().lower()
        raw_response = str(debug_payload.get("llm_raw_response", "")).strip()
        timeout_like = any(marker in validation_error for marker in ["timeout", "timed out", "read timed out"])
        empty_response = not raw_response
        if reason == "provider_error" and (timeout_like or empty_response):
            return True
        if validation_error == "invalid_json" and empty_response:
            return True
        return False

    for case_file in case_files:
        total += 1
        try:
            case_data = json.loads(case_file.read_text(encoding="utf-8"))
        except Exception as exc:
            failures.append(
                {
                    "name": case_file.stem,
                    "raw_stt": "",
                    "expected_draft_after": [],
                    "actual_draft_after": [],
                    "llm_raw_response": "",
                    "parsed_ops": [],
                    "validation_error": "invalid_case_json",
                    "reason": sanitize_text(str(exc), max_chars=200),
                }
            )
            continue

        name = sanitize_text(case_data.get("name", case_file.stem), max_chars=120)
        mode = sanitize_text(case_data.get("mode", default_mode or "correction"), max_chars=40).strip().lower() or (default_mode or "correction")
        draft_before = normalize_eval_blocks(case_data.get("draft_before", []))
        expected_draft_after = normalize_eval_blocks(case_data.get("expected_draft_after", []))
        expected_ops = case_data.get("expected_ops") if isinstance(case_data.get("expected_ops"), list) else []
        exact_expected_ops_match = bool(case_data.get("exact_expected_ops_match", False))
        raw_stt = sanitize_text(case_data.get("raw_stt", ""), max_chars=1200)
        context = case_data.get("context", {})
        if not isinstance(context, dict):
            context = {}

        retries_remaining = max(int(getattr(args, "retries", 1)), 0)
        attempts_used = 0
        while True:
            attempts_used += 1
            operation_result = plan_voice_operation(
                raw_stt,
                context,
                config,
                draft_before,
                forced_mode=mode,
            )
            if retries_remaining <= 0 or not should_retry_eval(operation_result):
                break
            retries_remaining -= 1
        actual_draft_after = normalize_eval_blocks(operation_result.get("draft_blocks", []))
        debug_payload = dict(operation_result.get("debug", {}) or {})
        debug_payload["eval_attempts"] = attempts_used
        operation_result["debug"] = debug_payload
        parsed_ops = debug_payload.get("parsed_ops", []) if isinstance(debug_payload.get("parsed_ops", []), list) else []
        ops_match = None
        if expected_ops:
            ops_match = parsed_ops == expected_ops
        passed_case = actual_draft_after == expected_draft_after and (not exact_expected_ops_match or bool(ops_match))
        if passed_case:
            passed += 1
        else:
            failures.append(
                {
                    "name": name,
                    "raw_stt": raw_stt,
                    "expected_draft_after": expected_draft_after,
                    "actual_draft_after": actual_draft_after,
                    "llm_raw_response": debug_payload.get("llm_raw_response", ""),
                    "parsed_ops": parsed_ops,
                    "validation_error": debug_payload.get("validation_error", ""),
                    "reason": operation_result.get("reason", ""),
                }
            )

        entry = build_eval_log_entry(
            case_name=name,
            raw_stt=raw_stt,
            draft_before=draft_before,
            expected_draft_after=expected_draft_after,
            operation_result=operation_result,
            config=config,
            mode=mode,
            exact_expected_ops_match=exact_expected_ops_match,
            expected_ops=expected_ops,
            passed=passed_case,
            ops_match=ops_match,
        )
        entry["replay_log_path"] = str(eval_log_file)
        save_replay_log_entry(config, entry, kind="eval", explicit_path=eval_log_file)

    print(f"total={total} passed={passed} failed={len(failures)}")
    for item in failures:
        print("")
        print(f"name: {item['name']}")
        print(f"raw_stt: {item['raw_stt']}")
        print(f"expected_draft_after: {json_dumps(item['expected_draft_after'])}")
        print(f"actual_draft_after: {json_dumps(item['actual_draft_after'])}")
        print(f"llm_raw_response: {item['llm_raw_response']}")
        print(f"parsed_ops: {json_dumps(item['parsed_ops'])}")
        print(f"validation_error: {item['validation_error']}")
        print(f"reason: {item['reason']}")
    if args.debug:
        print(f"eval_log: {eval_log_file}", file=sys.stderr)
    return 0 if not failures else 1


def command_eval_corrections(args: argparse.Namespace) -> int:
    return command_eval_cases(args, default_mode="correction")


def command_eval_append(args: argparse.Namespace) -> int:
    return command_eval_cases(args, default_mode="append")


def command_profile_current(args: argparse.Namespace) -> int:
    config = load_config()
    scope = get_scope_from_args(args, config, default_app="cli")
    profile = get_current_profile(config=config, scope=scope)
    status = get_context_status(config=config, scope=scope)
    print_profile_or_json(
        {
            "profile_id": profile.get("profile_id", scope["profile_id"]),
            "app": profile.get("app", scope["app"]),
            "session_id": profile.get("session_id", scope["session_id"]),
            "project_name": profile.get("project_name", scope["project_name"]),
            "project_root": profile.get("project_root", scope["project_root"]),
            "context_id": profile.get("context_id", scope["context_id"]),
            "last_active": profile.get("last_active", ""),
            "editor": editor_status(config, project_root=PROJECT_ROOT),
        },
        active_profile_id=str(scope["index"].get("active_profile_id", "")),
        args=args,
        status=status,
    )
    return 0


def command_profile_list(args: argparse.Namespace) -> int:
    config = load_config()
    scope = get_scope_from_args(args, config, default_app="cli")
    profiles = list_profiles(config=config, scope=scope)
    print_profile_list_or_json(
        profiles,
        active_profile_id=str(scope["index"].get("active_profile_id", "")),
        args=args,
    )
    return 0


def command_profile_use(args: argparse.Namespace) -> int:
    config = load_config()
    scope = get_scope_from_args(args, config, default_app="cli")
    try:
        profile = set_active_profile(args.profile_id, config=config, scope=scope)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    status = get_context_status(
        config=config,
        scope=resolve_context_scope(config=config, profile_id=args.profile_id, default_app="cli"),
    )
    print_profile_or_json(
        profile,
        active_profile_id=profile.get("profile_id", ""),
        args=args,
        status=status,
    )
    return 0


def command_profile_create(args: argparse.Namespace) -> int:
    config = load_config()
    entry = create_or_update_profile(
        profile_id=args.profile_id,
        app=args.app,
        session_id=args.session_id,
        project_root=args.project_root,
        context_id=args.context_id or "",
        config=config,
        set_active=False,
    )
    if args.json:
        print(json_dumps(entry))
    else:
        print(format_profile_compact(entry, active_profile_id="", include_updated=True))
    return 0


def command_profile_terms(args: argparse.Namespace) -> int:
    config = load_config()
    profile = get_profile(args.profile, config=config)
    if profile is None:
        print(f"未找到 profile: {args.profile}", file=sys.stderr)
        return 1
    scope = resolve_context_scope(
        config=config,
        profile_id=args.profile,
        default_app="cli",
    )
    full_context = load_full_context(config=config, scope=scope)
    print(json_dumps(full_context.get("term_bank", {})))
    return 0


def command_profile_switch(args: argparse.Namespace) -> int:
    config = load_config()
    scope = get_scope_from_args(args, config, default_app="cli")
    profiles = list_profiles(config=config, scope=scope)
    selected = prompt_profile_selection(
        profiles,
        active_profile_id=str(scope["index"].get("active_profile_id", "")),
    )
    if selected is None:
        return 0
    profile = set_active_profile(str(selected.get("profile_id", "")), config=config, scope=scope)
    print(
        "active profile: "
        + format_profile_compact(
            profile,
            active_profile_id=profile.get("profile_id", ""),
        ).lstrip("* ").lstrip()
    )
    return 0


def prompt_batch_switch_action(manager: VoiceQueueManager) -> str:
    if not manager.has_open_batch_work():
        return "switch"
    print("当前 batch 尚未结束：copy / discard / cancel switch [默认 copy]")
    choice = input("选择 [copy/discard/cancel]: ").strip().lower()
    if not choice:
        return "copy"
    if choice in {"copy", "discard", "cancel"}:
        return choice
    print("未识别输入，按 cancel 处理。", file=sys.stderr)
    return "cancel"


def command_listen(args: argparse.Namespace) -> int:
    config = deepcopy(load_config())
    apply_editor_overrides(config, args)
    config.setdefault("debug", {})
    config["debug"]["save_replay_logs"] = should_save_replay_logs(config, args)
    if args.hotkey:
        config.setdefault("audio", {})
        config["audio"]["hotkey"] = args.hotkey

    scope = get_scope_from_args(args, config, default_app="cli")

    audio_config = config.get("audio", {})
    recorder = AudioRecorder(
        sample_rate=int(audio_config.get("sample_rate", 16000)),
        channels=int(audio_config.get("channels", 1)),
        temp_dir=str(audio_config.get("temp_dir", "~/.voice_context/tmp_audio")),
    )
    manager = VoiceQueueManager(config=config, scope=scope, debug=args.debug, keep_audio=args.keep_audio)

    hotkey = str(audio_config.get("hotkey", "ctrl+alt+space"))
    correction_hotkey = str(audio_config.get("correction_hotkey", "c"))
    hold_to_record = bool(audio_config.get("hold_to_record", True))
    hotkey_monitor = None
    correction_hotkey_monitor = None
    hotkey_status = hotkey
    terminal_fallback = False
    active_recording_mode = ""

    def start_recording(requested_mode: str = "") -> None:
        nonlocal active_recording_mode
        try:
            if recorder.is_recording:
                return
            active_recording_mode = requested_mode
            recorder.start()
            manager.mark_recording(True)
        except (AudioDependencyError, RecordingError) as exc:
            manager.set_runtime_error(str(exc))
            print(str(exc), file=sys.stderr)

    def stop_recording() -> None:
        nonlocal active_recording_mode
        if not recorder.is_recording:
            return
        try:
            result = recorder.stop()
            manager.mark_recording(False)
            manager.enqueue_segment(result.audio_path, result.duration_sec, requested_mode=active_recording_mode)
            active_recording_mode = ""
        except RecordingError as exc:
            manager.mark_recording(False)
            manager.set_runtime_error(str(exc))
            print(str(exc), file=sys.stderr)

    if hold_to_record:
        try:
            hotkey_monitor = GlobalHotkeyMonitor(
                hotkey=hotkey,
                on_press=lambda: start_recording(""),
                on_release=stop_recording,
            )
            hotkey_monitor.start()
            correction_hotkey_monitor = GlobalHotkeyMonitor(
                hotkey=correction_hotkey,
                on_press=lambda: start_recording("correction"),
                on_release=stop_recording,
            )
            correction_hotkey_monitor.start()
        except HotkeyError as exc:
            hotkey_status = sanitize_text(str(exc), max_chars=140)
            terminal_fallback = True
            manager.set_runtime_error(f"{exc}；已切到 terminal 模式")
    else:
        terminal_fallback = True
        hotkey_status = "manual start/stop"

    manager.start_background_preload()

    last_version = -1
    last_render_at = 0.0
    ansi_enabled = supports_ansi_output()
    extra_lines: list[str] = []
    debug_visible = bool(args.debug)
    try:
        while True:
            manager.note_idle_rotation()
            snapshot = manager.get_snapshot()
            now = time.monotonic()
            dynamic_view = bool(recording_sec := recorder.recording_duration_sec) or str(snapshot.get("opt_state", "")) == "running" or any(
                segment.get("state") == "transcribing" for segment in snapshot.get("segments", [])
            )
            should_render = snapshot["version"] != last_version or (dynamic_view and now - last_render_at >= 0.2)
            if should_render:
                render_listen_snapshot(
                    snapshot,
                    debug_visible=debug_visible,
                    recording_sec=recording_sec,
                    ansi_enabled=ansi_enabled,
                    hotkey_status=f"append={hotkey} | correction={correction_hotkey}" if not terminal_fallback else hotkey_status,
                    terminal_fallback=terminal_fallback or hotkey_monitor is None,
                    correction_armed=bool(active_recording_mode == "correction"),
                    config=config,
                    extra_lines=extra_lines,
                )
                last_version = int(snapshot["version"])
                last_render_at = now
                if extra_lines:
                    extra_lines = []

            key = read_console_key()
            if key in {"enter", "space"} and hotkey_monitor is None:
                if recorder.is_recording:
                    stop_recording()
                else:
                    start_recording("")
            elif key == "c" and hotkey_monitor is None:
                if recorder.is_recording and active_recording_mode == "correction":
                    stop_recording()
                elif not recorder.is_recording:
                    start_recording("correction")
                extra_lines = ["correction recording"]
                last_version = -1
            elif key == "u":
                manager.undo_last_change()
            elif key == "x":
                manager.clear_current_batch()
            elif key == "n":
                manager.start_new_batch()
            elif key == "d":
                debug_visible = not debug_visible
                last_version = -1
            elif key == "q":
                break
            time.sleep(0.05)
    except KeyboardInterrupt:
        print("\n收到退出信号，正在结束 listen。", file=sys.stderr)
    finally:
        if recorder.is_recording:
            recorder.discard()
        if hotkey_monitor is not None:
            hotkey_monitor.stop()
        if correction_hotkey_monitor is not None:
            correction_hotkey_monitor.stop()
        manager.shutdown()
    return 0


def _tree_lines(root: Path) -> list[str]:
    ignored_dirs = {
        ".git",
        ".venv",
        "venv",
        "__pycache__",
        ".mypy_cache",
        ".pytest_cache",
        ".tmp_voice_context",
        ".voice_context",
        "local_samples",
        "tmp_audio",
        "logs",
        "transcripts",
        "debug_logs",
        "eval_runs",
    }
    lines: list[str] = []
    for path in sorted(root.rglob("*")):
        rel = path.relative_to(root)
        if any(part in ignored_dirs for part in rel.parts):
            continue
        depth = len(rel.parts) - 1
        suffix = "/" if path.is_dir() else ""
        lines.append("  " * depth + f"- {path.name}{suffix}")
    return lines


def command_structure(_: argparse.Namespace) -> int:
    structure_path = PROJECT_ROOT / "PROJECT_STRUCTURE.md"
    tree = "\n".join(_tree_lines(PROJECT_ROOT))
    content = f"""# PROJECT_STRUCTURE.md

## 项目结构

```text
voice-context-local/
{tree}
```

## 文件用途

- `README.md`：中英双语项目介绍、部署指南和使用指南。
- `config.yaml`：listen、STT、统一 editor/STT provider presets、语言上下文、clipboard 和运行时 prompt 模板配置。
- `secrets.example.yaml`：API key 本地配置模板；真实 `secrets.local.yaml` 不提交。
- `requirements.txt`：Python 依赖。
- `src/main.py`：CLI 入口，包含 listen、transcribe、status、structure 和 eval。
- `src/context_store.py`：session context、term_bank 和本地上下文存储。
- `src/prompt_editor.py`：listen 的 LLM append/correction、diff 和校验。
- `src/editor_providers.py`：Ollama 和 API preset 的 OpenAI-compatible 调用。
- `src/audio_recorder.py`：录音和热键。
- `src/voice_queue.py`：listen 队列、append/correction/undo、warmup 和 auto-copy。
- `src/stt_providers.py`：Qwen STT provider 分发。
- `src/qwen_stt_adapter.py`：STT 适配边界。
- `src/utils.py`：配置、API preset/secrets 解析、脱敏、剪贴板等通用工具。
- `tests/test_editor_api_config.py`：editor provider preset 配置解析、缺 key、mask、base_url 覆盖和未知 provider 校验。

## 常用命令

```bash
python src/main.py listen --editor openai --timeout 8 --debug
python src/main.py eval-append --cases eval_cases/append --editor openai --timeout 60 --debug
python src/main.py transcribe --audio "C:\\path\\to\\sample.wav" --session-id test-session-a --debug
python src/main.py structure
```
"""
    structure_path.write_text(content, encoding="utf-8")
    print(f"已生成：{structure_path}")
    return 0


def add_scope_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--session-id", default=None, help="优先使用的 session id")
    parser.add_argument("--context-id", default=None, help="备用 context id")
    parser.add_argument("--project-root", default=None, help="project root，用于 metadata 和 fallback hash")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Local voice context tool for AI coding workflows.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    show_parser = subparsers.add_parser("show", help="打印当前 voice context")
    add_scope_arguments(show_parser)
    show_parser.add_argument(
        "--mode",
        default="current",
        choices=["current", "asr", "optimizer", "full"],
        help="显示模式",
    )
    show_parser.set_defaults(func=command_show)

    status_parser = subparsers.add_parser("status", help="打印当前 context 路径和 session 状态")
    add_scope_arguments(status_parser)
    status_parser.add_argument("--json", action="store_true", help="输出完整 JSON")
    status_parser.set_defaults(func=command_status)

    clear_parser = subparsers.add_parser("clear", help="清空 voice context")
    add_scope_arguments(clear_parser)
    clear_parser.set_defaults(func=command_clear)

    update_parser = subparsers.add_parser("update", help="手动更新 voice context")
    add_scope_arguments(update_parser)
    update_parser.add_argument("--goal", default=None, help="当前目标")
    update_parser.add_argument("--dynamic", default=None, help="最新动态或修改要求")
    update_parser.add_argument("--ai-summary", default=None, help="最近 AI 摘要")
    update_parser.add_argument("--intent", default=None, help="最近用户意图")
    update_parser.add_argument("--terms", nargs="*", default=None, help="重要术语")
    update_parser.add_argument("--files", nargs="*", default=None, help="可能相关文件")
    update_parser.add_argument("--constraints", nargs="*", default=None, help="限制")
    update_parser.add_argument("--commands", nargs="*", default=None, help="下一步可能命令")
    update_parser.add_argument("--domain-terms", nargs="*", default=None, help="term_bank.domain_terms")
    update_parser.add_argument("--term-bank-files", nargs="*", default=None, help="term_bank.files")
    update_parser.add_argument("--term-bank-functions", nargs="*", default=None, help="term_bank.functions")
    update_parser.add_argument("--term-bank-classes", nargs="*", default=None, help="term_bank.classes")
    update_parser.add_argument("--term-bank-variables", nargs="*", default=None, help="term_bank.variables")
    update_parser.add_argument("--term-bank-commands", nargs="*", default=None, help="term_bank.commands")
    update_parser.add_argument("--term-bank-ui-terms", nargs="*", default=None, help="term_bank.ui_terms")
    update_parser.set_defaults(func=command_update)

    event_parser = subparsers.add_parser("event", help="追加一条 recent_messages 事件")
    add_scope_arguments(event_parser)
    event_parser.add_argument("--role", required=True, help="消息角色")
    event_parser.add_argument("--content", required=True, help="消息内容")
    event_parser.add_argument("--summary", default="", help="可选紧凑摘要")
    event_parser.set_defaults(func=command_event)

    transcribe_parser = subparsers.add_parser("transcribe", help="用 Qwen STT 转写本地音频文件")
    add_scope_arguments(transcribe_parser)
    transcribe_parser.add_argument("--audio", required=True, help="本地音频文件路径")
    transcribe_parser.add_argument("--debug", action="store_true", help="输出 STT 调试信息")
    transcribe_parser.set_defaults(func=command_transcribe)

    listen_parser = subparsers.add_parser("listen", help="启动常驻录音监听和 live batch queue")
    add_scope_arguments(listen_parser)
    listen_parser.add_argument(
        "--editor",
        choices=EDITOR_PROVIDER_CHOICES,
        default=None,
        help="临时覆盖 editor.provider preset",
    )
    listen_parser.add_argument("--model", default=None, help="临时覆盖当前 editor 的模型名")
    listen_parser.add_argument("--timeout", type=float, default=None, help="临时覆盖 editor 超时时间（秒）")
    listen_parser.add_argument("--hotkey", default=None, help="临时覆盖 audio.hotkey")
    listen_parser.add_argument("--keep-audio", action="store_true", help="保留临时录音文件")
    listen_parser.add_argument("--debug", action="store_true", help="显示 batch/segment 调试信息")
    listen_parser.add_argument("--save-debug-log", action="store_true", help="把 append/correction replay 详情写入 JSONL")
    listen_parser.set_defaults(func=command_listen)

    eval_parser = subparsers.add_parser("eval-corrections", help="运行 correction replay cases 并输出 pass/fail")
    eval_parser.add_argument("--cases", required=True, help="eval case 目录或单个 JSON 文件")
    eval_parser.add_argument(
        "--editor",
        choices=EDITOR_PROVIDER_CHOICES,
        default=None,
        help="临时覆盖 editor.provider preset",
    )
    eval_parser.add_argument("--model", default=None, help="临时覆盖当前 editor 的模型名")
    eval_parser.add_argument("--timeout", type=float, default=None, help="临时覆盖 editor 超时时间（秒）")
    eval_parser.add_argument("--retries", type=int, default=1, help="仅 provider timeout/空响应时的额外重试次数")
    eval_parser.add_argument("--debug", action="store_true", help="输出 eval run log 路径")
    eval_parser.set_defaults(func=command_eval_corrections)

    eval_append_parser = subparsers.add_parser("eval-append", help="运行 append replay cases 并输出 pass/fail")
    eval_append_parser.add_argument("--cases", required=True, help="eval case 目录或单个 JSON 文件")
    eval_append_parser.add_argument(
        "--editor",
        choices=EDITOR_PROVIDER_CHOICES,
        default=None,
        help="临时覆盖 editor.provider preset",
    )
    eval_append_parser.add_argument("--model", default=None, help="临时覆盖当前 editor 的模型名")
    eval_append_parser.add_argument("--timeout", type=float, default=None, help="临时覆盖 editor 超时时间（秒）")
    eval_append_parser.add_argument("--retries", type=int, default=1, help="仅 provider timeout/空响应时的额外重试次数")
    eval_append_parser.add_argument("--debug", action="store_true", help="输出 eval run log 路径")
    eval_append_parser.set_defaults(func=command_eval_append)

    structure_parser = subparsers.add_parser("structure", help="生成 PROJECT_STRUCTURE.md")
    structure_parser.set_defaults(func=command_structure)

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
