from __future__ import annotations

from typing import Any

try:
    from .stt_providers import STTError, transcribe_audio_with_meta
except ImportError:
    from stt_providers import STTError, transcribe_audio_with_meta


def transcribe_with_meta(
    audio_path: str,
    asr_context: str = "",
    *,
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return transcribe_audio_with_meta(audio_path, asr_context, config=config)


def transcribe(audio_path: str, asr_context: str = "") -> str:
    """Adapter boundary for the project's STT provider stack.

    - audio_path: local audio file path.
    - asr_context: compact local ASR context / term hints.
    - return: raw STT text.
    """
    return transcribe_with_meta(audio_path, asr_context)["text"]
