from __future__ import annotations

import queue
import threading
import time
from dataclasses import dataclass, field
from typing import Any

try:
    from .audio_recorder import cleanup_audio_file
    from .context_store import context_for_asr, load_full_context, resolve_context_scope
    from .editor_providers import ProviderError, warmup_ollama
    from .prompt_editor import VOICE_OP_IGNORE, VOICE_OP_UNDO, detect_voice_command, plan_voice_operation
    from .qwen_stt_adapter import STTError, transcribe_with_meta
    from .stt_providers import warmup_stt_provider
    from .utils import PROJECT_ROOT, copy_to_clipboard, editor_provider_name, replay_log_path, sanitize_text, save_replay_log_entry, utc_now_iso
except ImportError:
    from audio_recorder import cleanup_audio_file
    from context_store import context_for_asr, load_full_context, resolve_context_scope
    from editor_providers import ProviderError, warmup_ollama
    from prompt_editor import VOICE_OP_IGNORE, VOICE_OP_UNDO, detect_voice_command, plan_voice_operation
    from qwen_stt_adapter import STTError, transcribe_with_meta
    from stt_providers import warmup_stt_provider
    from utils import PROJECT_ROOT, copy_to_clipboard, editor_provider_name, replay_log_path, sanitize_text, save_replay_log_entry, utc_now_iso


@dataclass
class SegmentRecord:
    segment_id: int
    audio_path: str
    duration_sec: float
    requested_mode: str = ""
    raw_stt: str = ""
    state: str = "queued"
    error: str = ""
    stt_elapsed_ms: int = 0
    started_monotonic: float = 0.0
    finished_monotonic: float = 0.0
    operation: str = ""


@dataclass
class ChangeRecord:
    generation: int
    operation: str
    segment_id: int
    previous_blocks: list[dict[str, Any]]
    new_blocks: list[dict[str, Any]]
    diff_items: list[dict[str, Any]]
    copied_text: str
    fallback: bool = False
    fallback_reason: str = ""
    source_text: str = ""


@dataclass
class BatchRecord:
    batch_id: int
    profile_id: str
    app: str
    session_id: str
    context_id: str
    created_at: str
    updated_at: str
    closed_at: str = ""
    segments: list[SegmentRecord] = field(default_factory=list)
    draft_blocks: list[dict[str, Any]] = field(default_factory=list)
    change_history: list[ChangeRecord] = field(default_factory=list)
    latest_prompt: str = ""
    latest_generation: int = 0
    copied_generation: int = 0
    pending_generation: int = 0
    state: str = "idle"
    last_error: str = ""
    last_copy_error: str = ""
    editor_elapsed_ms: int = 0
    fallback: bool = False
    suppress_processing: bool = False
    latest_fallback_reason: str = ""
    short_test_input: bool = False
    skipped_project_prompt: bool = False
    current_diff: list[dict[str, Any]] = field(default_factory=list)
    current_change_title: str = "Current change:"
    current_change_generation: int = 0
    current_change_op: str = ""
    current_change_reason: str = ""
    current_change_source: str = ""
    current_change_debug: dict[str, Any] = field(default_factory=dict)
    draft_updated_at_monotonic: float = 0.0


def format_profile_compact(
    profile: dict[str, Any],
    *,
    active_profile_id: str = "",
    index: int | None = None,
    include_updated: bool = False,
) -> str:
    marker = "* " if profile.get("profile_id", "") == active_profile_id else "  "
    prefix = f"{index}. " if index is not None else ""
    profile_id = profile.get("profile_id", "") or "default-coding"
    app = profile.get("app", "") or "default"
    session_id = profile.get("session_id", "") or "default"
    project_name = profile.get("project_name", "") or profile.get("context_id", "") or "default-coding"
    context_id = profile.get("context_id", "") or session_id
    line = f"{marker}{prefix}{profile_id} | {app}/{session_id} | {project_name} | ctx={context_id}"
    last_active = str(profile.get("last_active", "")).replace("T", " ").replace("+00:00", " UTC")
    if include_updated and last_active:
        line += f" | updated {last_active[:19]}"
    return line


def format_status_compact(status: dict[str, Any]) -> str:
    profile_id = status.get("profile_id", "") or "default-coding"
    app = status.get("app", "") or "default"
    session_id = status.get("session_id", "") or "default"
    context_id = status.get("context_id", "") or session_id
    context_path = status.get("context_path", "")
    return f"{profile_id} | {app}/{session_id} | ctx={context_id} | {context_path}"


def _copy_blocks(blocks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [{"id": int(block.get("id", 0)), "text": str(block.get("text", ""))} for block in blocks]


def _reindex_blocks(blocks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    reindexed: list[dict[str, Any]] = []
    for index, block in enumerate(blocks, start=1):
        text = sanitize_text(str(block.get("text", "")).strip(), max_chars=400).strip()
        if not text:
            continue
        reindexed.append({"id": index, "text": text})
    return reindexed


def _join_blocks(blocks: list[dict[str, Any]]) -> str:
    return " ".join(
        sanitize_text(str(block.get("text", "")).strip(), max_chars=400).strip()
        for block in blocks
        if str(block.get("text", "")).strip()
    ).strip()


class VoiceQueueManager:
    def __init__(
        self,
        *,
        config: dict[str, Any],
        scope: dict[str, Any],
        debug: bool = False,
        keep_audio: bool = False,
    ) -> None:
        self.config = config
        self.debug = debug
        self.keep_audio = keep_audio
        voice_queue_config = config.get("voice_queue", {})
        self.edit_model = str(voice_queue_config.get("edit_model", "append_patch")).strip().lower() or "append_patch"
        self.auto_rotate = bool(voice_queue_config.get("auto_rotate", self.edit_model != "append_patch"))
        raw_merge_window_sec = voice_queue_config.get("merge_window_sec", 300)
        raw_max_segments_per_batch = voice_queue_config.get("max_segments_per_batch", 20)
        self.merge_window_sec = 0.0 if raw_merge_window_sec in {None, 0, "", False} else max(1.0, float(raw_merge_window_sec))
        self.max_segments_per_batch = 0 if raw_max_segments_per_batch in {None, 0, "", False} else max(1, int(raw_max_segments_per_batch))
        self.stt_workers = max(1, int(voice_queue_config.get("stt_workers", 1)))
        self.live_clipboard_update = bool(voice_queue_config.get("live_clipboard_update", True))
        self.copy_policy = str(voice_queue_config.get("copy_policy", "latest_only"))
        debug_config = config.get("debug", {}) if isinstance(config.get("debug", {}), dict) else {}
        self.save_replay_logs = bool(debug_config.get("save_replay_logs", False))

        self._scope = dict(scope)
        self._lock = threading.RLock()
        self._version = 0
        self._stop_event = threading.Event()
        self._stt_queue: queue.Queue[tuple[int, int]] = queue.Queue()
        self._op_queue: queue.Queue[tuple[int, int, int]] = queue.Queue()
        self._batches: dict[int, BatchRecord] = {}
        self._active_batch_id = 0
        self._next_batch_id = 1
        self._next_segment_id = 1
        self._last_clipboard_message = ""
        self._state_message = "idle"
        self._active_recording = False
        self._stt_warm_state = "idle"
        self._stt_warm_message = ""
        self._editor_warm_state = "idle"
        self._editor_warm_message = ""

        self._create_new_batch_locked(scope=self._scope)

        self._stt_threads = [
            threading.Thread(target=self._stt_loop, name=f"voice-queue-stt-{index+1}", daemon=True)
            for index in range(self.stt_workers)
        ]
        self._op_thread = threading.Thread(target=self._operation_loop, name="voice-queue-operation", daemon=True)
        for thread in self._stt_threads:
            thread.start()
        self._op_thread.start()

    @property
    def scope(self) -> dict[str, Any]:
        with self._lock:
            return dict(self._scope)

    def mark_recording(self, is_recording: bool) -> None:
        with self._lock:
            self._active_recording = is_recording
            self._bump_version_locked()

    def set_warmup_state(self, kind: str, state: str, message: str = "") -> None:
        with self._lock:
            clean_state = sanitize_text(state, max_chars=40) or "idle"
            clean_message = sanitize_text(message, max_chars=160)
            if kind == "stt":
                self._stt_warm_state = clean_state
                self._stt_warm_message = clean_message
            elif kind == "editor":
                self._editor_warm_state = clean_state
                self._editor_warm_message = clean_message
            self._bump_version_locked()

    def start_background_preload(self) -> None:
        queue_config = self.config.get("voice_queue", {})
        preload_stt = bool(queue_config.get("preload_stt_on_start", True))
        preload_editor = bool(queue_config.get("preload_editor_on_start", True))

        if preload_stt:
            self.set_warmup_state("stt", "warming", "warming qwen stt")

            def run_stt_warmup() -> None:
                try:
                    result = warmup_stt_provider(config=self.config)
                    self.set_warmup_state("stt", "ready", str(result.get("message", "stt ready")))
                except Exception as exc:
                    self.set_warmup_state("stt", "error", str(exc))

            threading.Thread(target=run_stt_warmup, name="voice-queue-stt-warmup", daemon=True).start()
        else:
            self.set_warmup_state("stt", "idle", "")

        editor_mode = editor_provider_name(self.config)
        if preload_editor and editor_mode == "ollama":
            self.set_warmup_state("editor", "warming", "warming ollama editor")

            def run_editor_warmup() -> None:
                try:
                    timeout_sec = float(self.config.get("editor", {}).get("timeout_sec", 2.0))
                    result = warmup_ollama(config=self.config, timeout_sec=max(timeout_sec, 8.0))
                    self.set_warmup_state("editor", "ready", str(result.get("message", "editor ready")))
                except ProviderError as exc:
                    self.set_warmup_state("editor", "error", str(exc))
                except Exception as exc:
                    self.set_warmup_state("editor", "error", str(exc))

            threading.Thread(target=run_editor_warmup, name="voice-queue-editor-warmup", daemon=True).start()
        else:
            self.set_warmup_state("editor", "idle", "")

    def set_runtime_error(self, message: str) -> None:
        with self._lock:
            self._state_message = f"error: {sanitize_text(message, max_chars=140)}"
            self._bump_version_locked()

    def has_open_batch_work(self) -> bool:
        with self._lock:
            batch = self._batches[self._active_batch_id]
            return bool(batch.segments or batch.draft_blocks or batch.latest_prompt)

    def shutdown(self) -> None:
        self._stop_event.set()
        for _ in range(self.stt_workers):
            self._stt_queue.put((-1, -1))
        self._op_queue.put((-1, -1, -1))
        for thread in self._stt_threads:
            thread.join(timeout=2.0)
        self._op_thread.join(timeout=2.0)

    def enqueue_segment(self, audio_path: str, duration_sec: float, *, requested_mode: str = "") -> int:
        with self._lock:
            if self.auto_rotate:
                self._maybe_rotate_batch_locked(reason="enqueue")
            batch = self._batches[self._active_batch_id]
            segment = SegmentRecord(
                segment_id=self._next_segment_id,
                audio_path=audio_path,
                duration_sec=duration_sec,
                requested_mode=requested_mode,
            )
            self._next_segment_id += 1
            batch.segments.append(segment)
            batch.state = f"queued seg={len(batch.segments)}"
            batch.updated_at = utc_now_iso()
            self._state_message = batch.state
            self._bump_version_locked()
            batch_id = batch.batch_id
            segment_id = segment.segment_id
        self._stt_queue.put((batch_id, segment_id))
        return segment_id

    def clear_current_batch(self) -> None:
        with self._lock:
            self._close_active_batch_locked(reason="discard", suppress_processing=True)
            self._create_new_batch_locked(scope=self._scope)
            self._state_message = "idle"
            self._bump_version_locked()

    def start_new_batch(self) -> None:
        with self._lock:
            self._close_active_batch_locked(reason="new", suppress_processing=False)
            self._create_new_batch_locked(scope=self._scope)
            self._state_message = "idle"
            self._bump_version_locked()

    def undo_last_change(self, *, source: str = "keyboard", batch_id: int | None = None) -> bool:
        with self._lock:
            target_batch_id = batch_id if batch_id is not None else self._active_batch_id
            batch = self._batches[target_batch_id]
            if not batch.change_history:
                batch.state = f"undo skipped ({source})"
                batch.current_change_title = "UNDO:"
                batch.current_change_generation = batch.latest_generation
                batch.current_change_source = f"{source}: 撤销"
                batch.current_diff = [{"kind": "undo", "generation": "", "operation": "noop"}]
                self._state_message = batch.state
                self._bump_version_locked()
                return False
            last_change = batch.change_history.pop()
            batch.draft_blocks = _reindex_blocks(last_change.previous_blocks)
            batch.latest_prompt = _join_blocks(batch.draft_blocks)
            batch.latest_generation += 1
            generation = batch.latest_generation
            batch.pending_generation = 0
            batch.current_diff = [
                {
                    "kind": "undo",
                    "generation": last_change.generation,
                    "operation": last_change.operation,
                }
            ]
            batch.current_change_title = "UNDO:"
            batch.current_change_generation = generation
            batch.current_change_op = VOICE_OP_UNDO
            batch.current_change_reason = ""
            batch.current_change_source = f"{source}: 撤销"
            batch.current_change_debug = {"mode": "undo", "source": source}
            batch.editor_elapsed_ms = 0
            batch.fallback = False
            batch.latest_fallback_reason = ""
            batch.short_test_input = False
            batch.skipped_project_prompt = False
            batch.updated_at = utc_now_iso()
            copied = False
            if batch.latest_prompt and self.live_clipboard_update and self.copy_policy == "latest_only":
                ok, message = copy_to_clipboard(batch.latest_prompt)
                if ok:
                    batch.copied_generation = generation
                    batch.last_copy_error = ""
                    self._last_clipboard_message = message
                    copied = True
                else:
                    batch.last_copy_error = sanitize_text(message, max_chars=160)
            batch.state = f"{'copied' if copied else ('copy error' if batch.last_copy_error else 'done')} gen={generation}"
            self._state_message = batch.state
            self._bump_version_locked()
            return True

    def set_scope(self, scope: dict[str, Any], *, copy_latest_prompt_first: bool = False, discard_open_batch: bool = False) -> None:
        if copy_latest_prompt_first:
            self.copy_latest_prompt()
        with self._lock:
            self._close_active_batch_locked(reason="switch", suppress_processing=discard_open_batch)
            self._scope = dict(scope)
            self._create_new_batch_locked(scope=self._scope)
            self._state_message = "idle"
            self._bump_version_locked()

    def copy_latest_prompt(self) -> tuple[bool, str]:
        with self._lock:
            batch = self._batches[self._active_batch_id]
            prompt = batch.latest_prompt
        if not prompt:
            return False, "当前 batch 还没有可复制的 prompt。"
        ok, message = copy_to_clipboard(prompt)
        with self._lock:
            batch = self._batches[self._active_batch_id]
            if ok:
                batch.copied_generation = max(batch.copied_generation, batch.latest_generation)
                batch.last_copy_error = ""
                batch.state = f"copied gen={batch.latest_generation}"
                self._state_message = batch.state
                self._last_clipboard_message = message
                self._bump_version_locked()
            else:
                batch.last_copy_error = sanitize_text(message, max_chars=160)
                batch.state = f"copy error gen={batch.latest_generation}"
                self._state_message = batch.state
                self._bump_version_locked()
        return ok, message

    def get_snapshot(self) -> dict[str, Any]:
        with self._lock:
            batch = self._batches[self._active_batch_id]
            now_monotonic = time.monotonic()
            current_segment: dict[str, Any] | None = None
            segments: list[dict[str, Any]] = []
            done_segments = 0
            for segment in batch.segments:
                preview_text = segment.raw_stt or segment.error or "(waiting)"
                if segment.state == "done":
                    done_segments += 1
                elapsed_sec = (
                    round(max(0.0, now_monotonic - segment.started_monotonic), 1)
                    if segment.state == "transcribing" and segment.started_monotonic
                    else 0.0
                )
                if segment.state in {"queued", "transcribing", "error"} and current_segment is None:
                    current_segment = {
                        "segment_id": segment.segment_id,
                        "state": segment.state,
                        "elapsed_sec": elapsed_sec,
                        "error": sanitize_text(segment.error, max_chars=140),
                    }
                segments.append(
                    {
                        "segment_id": segment.segment_id,
                        "preview": sanitize_text(preview_text, max_chars=110),
                        "state": segment.state,
                        "requested_mode": segment.requested_mode,
                        "operation": segment.operation,
                        "stt_elapsed_ms": segment.stt_elapsed_ms,
                        "elapsed_sec": elapsed_sec,
                        "error": sanitize_text(segment.error, max_chars=140),
                    }
                )
            total_segments = len(batch.segments)
            stt_state = (
                f"#{current_segment.get('segment_id')} {current_segment.get('elapsed_sec', 0):.1f}s"
                if current_segment and current_segment.get("state") == "transcribing"
                else ("idle" if total_segments == 0 else f"{done_segments}/{total_segments}")
            )
            opt_state = "running" if batch.state.startswith("optimizing") else "done"
            if current_segment and current_segment.get("state") == "queued":
                opt_state = "pending"
            if batch.latest_generation <= 0:
                copy_state = "none"
            elif batch.copied_generation >= batch.latest_generation:
                copy_state = f"gen={batch.copied_generation} copied"
            elif batch.last_copy_error:
                copy_state = "error"
            else:
                copy_state = "pending"
            return {
                "version": self._version,
                "profile_id": batch.profile_id,
                "app": batch.app,
                "session_id": batch.session_id,
                "context_id": batch.context_id,
                "batch_id": batch.batch_id,
                "generation": batch.latest_generation,
                "copied_generation": batch.copied_generation,
                "pending_generation": batch.pending_generation,
                "state": self._state_message,
                "segments": segments,
                "done_segments": done_segments,
                "total_segments": total_segments,
                "current_segment": current_segment,
                "recording": self._active_recording,
                "stt_state": stt_state,
                "opt_state": opt_state,
                "copy_state": copy_state,
                "latest_prompt_preview": sanitize_text(batch.latest_prompt, max_chars=500),
                "draft_blocks": _copy_blocks(batch.draft_blocks),
                "current_diff": list(batch.current_diff),
                "current_change_title": batch.current_change_title,
                "current_change_generation": batch.current_change_generation,
                "current_change_op": batch.current_change_op,
                "current_change_reason": batch.current_change_reason,
                "current_change_source": sanitize_text(batch.current_change_source, max_chars=200),
                "current_change_debug": dict(batch.current_change_debug),
                "editor_elapsed_ms": batch.editor_elapsed_ms,
                "fallback": batch.fallback,
                "fallback_reason": batch.latest_fallback_reason,
                "short_test_input": batch.short_test_input,
                "skipped_project_prompt": batch.skipped_project_prompt,
                "stt_warm_state": self._stt_warm_state,
                "stt_warm_message": self._stt_warm_message,
                "editor_warm_state": self._editor_warm_state,
                "editor_warm_message": self._editor_warm_message,
                "clipboard_message": self._last_clipboard_message,
                "copy_error": batch.last_copy_error,
            }

    def note_idle_rotation(self) -> None:
        if not self.auto_rotate:
            return
        with self._lock:
            self._maybe_rotate_batch_locked(reason="idle")

    def _bump_version_locked(self) -> None:
        self._version += 1

    def _create_new_batch_locked(self, *, scope: dict[str, Any]) -> BatchRecord:
        now = utc_now_iso()
        batch = BatchRecord(
            batch_id=self._next_batch_id,
            profile_id=str(scope.get("profile_id", "")),
            app=str(scope.get("app", "")),
            session_id=str(scope.get("session_id", "")),
            context_id=str(scope.get("context_id", "")),
            created_at=now,
            updated_at=now,
        )
        self._next_batch_id += 1
        self._batches[batch.batch_id] = batch
        self._active_batch_id = batch.batch_id
        return batch

    def _close_active_batch_locked(self, *, reason: str, suppress_processing: bool) -> None:
        batch = self._batches[self._active_batch_id]
        batch.closed_at = utc_now_iso()
        batch.suppress_processing = suppress_processing
        if not batch.state.startswith("error"):
            batch.state = f"closed:{reason}"

    def _maybe_rotate_batch_locked(self, *, reason: str) -> None:
        if not self.auto_rotate:
            return
        batch = self._batches[self._active_batch_id]
        if not batch.segments and not batch.draft_blocks:
            return
        if self._active_recording:
            return
        if any(segment.state in {"queued", "transcribing"} for segment in batch.segments):
            return
        if batch.state.startswith("optimizing"):
            return
        if batch.latest_prompt and batch.copied_generation < batch.latest_generation:
            return
        idle_sec = max(0.0, time.time() - self._iso_to_ts(batch.updated_at))
        over_max_segments = self.max_segments_per_batch > 0 and len(batch.segments) >= self.max_segments_per_batch
        over_merge_window = self.merge_window_sec > 0 and idle_sec > self.merge_window_sec
        if over_max_segments or over_merge_window:
            self._close_active_batch_locked(reason=reason, suppress_processing=False)
            self._create_new_batch_locked(scope=self._scope)
            self._state_message = "idle"
            self._bump_version_locked()

    def _iso_to_ts(self, value: str) -> float:
        if not value:
            return time.time()
        try:
            from datetime import datetime

            return datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp()
        except Exception:
            return time.time()

    def _current_scope_for_batch(self, batch: BatchRecord) -> dict[str, Any]:
        return resolve_context_scope(
            config=self.config,
            profile_id=batch.profile_id,
            session_id=batch.session_id,
            context_id=batch.context_id,
            app=batch.app,
            default_app=batch.app,
        )

    def _save_replay_log(
        self,
        *,
        batch: BatchRecord,
        segment: SegmentRecord,
        attempted_generation: int,
        mode: str,
        forced_mode: str,
        draft_before: list[dict[str, Any]],
        operation_result: dict[str, Any],
    ) -> None:
        if not self.save_replay_logs:
            return
        debug_payload = dict(operation_result.get("debug", {}) or {})
        python_validation = {
            "accepted": not bool(operation_result.get("ignored", False)),
            "ignored": bool(operation_result.get("ignored", False)),
            "fallback": bool(operation_result.get("fallback", False)),
            "validation_error": str(debug_payload.get("validation_error", "")).strip(),
            "fallback_reason": str(operation_result.get("fallback_reason", "")).strip(),
            "normalization_used": bool(debug_payload.get("normalization_used", False)),
        }
        entry = {
            "case_id": f"listen-b{batch.batch_id}-s{segment.segment_id}-g{attempted_generation}",
            "timestamp": utc_now_iso(),
            "generation": attempted_generation,
            "mode": mode,
            "forced_mode": forced_mode,
            "prompt_version": str(debug_payload.get("prompt_version", "")),
            "prompt_variant": str(debug_payload.get("prompt_variant", "")),
            "raw_stt": segment.raw_stt,
            "draft_before": _copy_blocks(draft_before),
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
            "draft_after": _copy_blocks(operation_result.get("draft_blocks", [])),
            "diff_items": list(operation_result.get("diff_items", [])),
            "reason": str(operation_result.get("reason", "")).strip(),
            "editor_elapsed_ms": int(operation_result.get("editor_elapsed_ms", 0) or 0),
            "provider": debug_payload.get("provider", {}),
            "config_path": str(PROJECT_ROOT / "config.yaml"),
            "replay_log_path": str(replay_log_path(self.config, kind="debug")),
            "eval_case_name": "",
        }
        ok, message = save_replay_log_entry(self.config, entry, kind="debug")
        if not ok:
            batch.last_error = f"debug log failed: {message}"

    def _stt_loop(self) -> None:
        while not self._stop_event.is_set():
            batch_id, segment_id = self._stt_queue.get()
            if batch_id < 0 or segment_id < 0:
                return
            with self._lock:
                batch = self._batches.get(batch_id)
                if batch is None:
                    continue
                segment = next((item for item in batch.segments if item.segment_id == segment_id), None)
                if segment is None:
                    continue
                if batch.suppress_processing and batch.closed_at:
                    segment.state = "discarded"
                    self._bump_version_locked()
                    continue
                segment.state = "transcribing"
                segment.started_monotonic = time.monotonic()
                batch.state = f"transcribing seg={segment.segment_id}/{len(batch.segments)}"
                self._state_message = batch.state
                self._bump_version_locked()
                scope = self._current_scope_for_batch(batch)
            try:
                full_context = load_full_context(config=self.config, scope=scope)
                asr_context = context_for_asr(full_context, config=self.config, scope=scope)
                stt_result = transcribe_with_meta(segment.audio_path, asr_context, config=self.config)
            except STTError as exc:
                with self._lock:
                    segment.state = "error"
                    segment.error = sanitize_text(str(exc), max_chars=200)
                    batch.state = f"error: {segment.error}"
                    batch.last_error = segment.error
                    batch.updated_at = utc_now_iso()
                    self._state_message = batch.state
                    self._bump_version_locked()
                continue
            finally:
                if not self.keep_audio:
                    cleanup_audio_file(segment.audio_path)

            with self._lock:
                latest = self._batches.get(batch_id)
                if latest is None:
                    continue
                live_segment = next((item for item in latest.segments if item.segment_id == segment_id), None)
                if live_segment is None:
                    continue
                live_segment.raw_stt = sanitize_text(stt_result.get("text", ""), max_chars=1200)
                live_segment.stt_elapsed_ms = int(stt_result.get("stt_elapsed_ms", 0))
                live_segment.state = "done"
                live_segment.finished_monotonic = time.monotonic()
                latest.pending_generation = latest.latest_generation + 1
                latest.updated_at = utc_now_iso()
                latest.state = f"queued seg={len(latest.segments)}"
                self._state_message = latest.state
                self._bump_version_locked()
            self._op_queue.put((batch_id, segment_id, latest.pending_generation))

    def _apply_operation_locked(
        self,
        batch: BatchRecord,
        segment_id: int,
        operation_result: dict[str, Any],
    ) -> None:
        op = str(operation_result.get("op", "append"))
        fallback = bool(operation_result.get("fallback", False))
        fallback_reason = str(operation_result.get("fallback_reason", "")).strip()
        draft_blocks = _copy_blocks(operation_result.get("draft_blocks", []))
        diff_items = list(operation_result.get("diff_items", []))
        prompt = str(operation_result.get("prompt", "")).strip() or _join_blocks(draft_blocks)
        debug_payload = dict(operation_result.get("debug", {}) or {})

        batch.editor_elapsed_ms = int(operation_result.get("editor_elapsed_ms", 0) or 0)
        batch.fallback = fallback
        batch.latest_fallback_reason = fallback_reason
        batch.short_test_input = False
        batch.skipped_project_prompt = bool(operation_result.get("ignored", False))
        batch.current_diff = diff_items
        batch.current_change_title = "UNDO:" if op == VOICE_OP_UNDO else "Current change:"
        batch.current_change_source = str(operation_result.get("raw_text", "")).strip()
        batch.current_change_op = op
        ignored_reason = str(operation_result.get("reason", "")).strip()
        batch.current_change_reason = ignored_reason
        batch.current_change_debug = debug_payload
        batch.pending_generation = 0

        if op == VOICE_OP_IGNORE:
            batch.current_change_title = "Correction ignored:" if debug_payload.get("correction_raw_text") else "Ignored:"
            batch.current_change_generation = batch.latest_generation
            batch.latest_fallback_reason = ignored_reason
            batch.state = f"ignored {ignored_reason or 'segment'} gen={batch.latest_generation}"
            self._state_message = batch.state
            batch.updated_at = utc_now_iso()
            self._bump_version_locked()
            return

        generation = batch.latest_generation + 1
        batch.latest_generation = generation
        batch.current_change_generation = generation
        previous_blocks = _copy_blocks(batch.draft_blocks)
        batch.draft_blocks = _reindex_blocks(draft_blocks)
        batch.latest_prompt = prompt
        batch.updated_at = utc_now_iso()

        copied = False
        if batch.latest_prompt and self.live_clipboard_update and self.copy_policy == "latest_only":
            ok, message = copy_to_clipboard(batch.latest_prompt)
            if ok:
                batch.copied_generation = generation
                batch.last_copy_error = ""
                self._last_clipboard_message = message
                copied = True
            else:
                batch.last_copy_error = sanitize_text(message, max_chars=160)

        batch.change_history.append(
            ChangeRecord(
                generation=generation,
                operation=op,
                segment_id=segment_id,
                previous_blocks=previous_blocks,
                new_blocks=_copy_blocks(batch.draft_blocks),
                diff_items=list(diff_items),
                copied_text=batch.latest_prompt,
                fallback=fallback,
                fallback_reason=fallback_reason,
                source_text=batch.current_change_source,
            )
        )
        batch.state = f"{'copied' if copied else ('copy error' if batch.last_copy_error else 'done')} gen={generation}"
        self._state_message = batch.state
        self._bump_version_locked()

    def _operation_loop(self) -> None:
        while not self._stop_event.is_set():
            batch_id, segment_id, _requested_generation = self._op_queue.get()
            if batch_id < 0:
                return
            with self._lock:
                batch = self._batches.get(batch_id)
                if batch is None or batch.suppress_processing:
                    continue
                segment = next((item for item in batch.segments if item.segment_id == segment_id), None)
                if segment is None or not segment.raw_stt.strip():
                    continue
                pending_generation = batch.latest_generation + 1
                batch.pending_generation = pending_generation
                batch.state = f"optimizing gen={pending_generation}"
                self._state_message = batch.state
                self._bump_version_locked()
                scope = self._current_scope_for_batch(batch)
                draft_blocks = _copy_blocks(batch.draft_blocks)
                requested_mode = segment.requested_mode
                segment_text = segment.raw_stt
            try:
                full_context = load_full_context(config=self.config, scope=scope)
                command_result = detect_voice_command(segment_text, self.config, forced_mode="")
                if str(command_result.get("intent", "")) == "undo":
                    self.undo_last_change(source="voice", batch_id=batch_id)
                    with self._lock:
                        latest = self._batches.get(batch_id)
                        if latest is not None:
                            live_segment = next((item for item in latest.segments if item.segment_id == segment_id), None)
                            if live_segment is not None:
                                live_segment.operation = VOICE_OP_UNDO
                    continue
                operation_result = plan_voice_operation(
                    segment_text,
                    full_context,
                    self.config,
                    draft_blocks,
                    forced_mode=requested_mode,
                )
                planned_mode = "correction" if requested_mode == "correction" else str(command_result.get("intent", "")).strip().lower()
            except Exception as exc:
                with self._lock:
                    latest = self._batches.get(batch_id)
                    if latest is None:
                        continue
                    latest.state = f"error: {sanitize_text(str(exc), max_chars=160)}"
                    latest.last_error = sanitize_text(str(exc), max_chars=160)
                    latest.latest_fallback_reason = latest.last_error
                    self._state_message = latest.state
                    self._bump_version_locked()
                continue

            with self._lock:
                latest = self._batches.get(batch_id)
                if latest is None or latest.suppress_processing:
                    continue
                live_segment = next((item for item in latest.segments if item.segment_id == segment_id), None)
                if live_segment is None:
                    continue
                live_segment.operation = str(operation_result.get("op", ""))
                if planned_mode in {"append", "correction"}:
                    self._save_replay_log(
                        batch=latest,
                        segment=live_segment,
                        attempted_generation=pending_generation,
                        mode=planned_mode,
                        forced_mode=requested_mode,
                        draft_before=draft_blocks,
                        operation_result=operation_result,
                    )
                self._apply_operation_locked(latest, segment_id, operation_result)
