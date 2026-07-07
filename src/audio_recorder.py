from __future__ import annotations

import threading
import time
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

try:
    from .utils import expand_user_path, utc_now_iso
except ImportError:
    from utils import expand_user_path, utc_now_iso


class AudioDependencyError(RuntimeError):
    pass


class RecordingError(RuntimeError):
    pass


class HotkeyError(RuntimeError):
    pass


def _load_audio_dependencies():
    try:
        import numpy  # type: ignore
        import sounddevice  # type: ignore
    except ImportError as exc:
        raise AudioDependencyError(
            "录音功能缺少依赖，请运行：python -m pip install sounddevice numpy"
        ) from exc
    return numpy, sounddevice


@dataclass
class RecordingResult:
    audio_path: str
    duration_sec: float


class AudioRecorder:
    def __init__(
        self,
        *,
        sample_rate: int,
        channels: int,
        temp_dir: str,
    ) -> None:
        self.sample_rate = int(sample_rate)
        self.channels = int(channels)
        self.temp_dir = expand_user_path(temp_dir)
        self.temp_dir.mkdir(parents=True, exist_ok=True)
        self._numpy = None
        self._sounddevice = None
        self._stream = None
        self._chunks = []
        self._started_at = 0.0
        self._lock = threading.Lock()
        self._recording = False

    @property
    def is_recording(self) -> bool:
        with self._lock:
            return self._recording

    @property
    def recording_duration_sec(self) -> float:
        with self._lock:
            if not self._recording or not self._started_at:
                return 0.0
            return max(0.0, time.perf_counter() - self._started_at)

    def start(self) -> None:
        with self._lock:
            if self._recording:
                return
            numpy, sounddevice = _load_audio_dependencies()
            self._numpy = numpy
            self._sounddevice = sounddevice
            self._chunks = []

            def callback(indata, frames, time_info, status):  # type: ignore[no-untyped-def]
                _ = frames, time_info, status
                self._chunks.append(indata.copy())

            try:
                self._stream = sounddevice.InputStream(
                    samplerate=self.sample_rate,
                    channels=self.channels,
                    dtype="int16",
                    callback=callback,
                )
                self._stream.start()
            except Exception as exc:
                self._stream = None
                raise RecordingError(f"无法开始录音：{exc}") from exc

            self._started_at = time.perf_counter()
            self._recording = True

    def stop(self) -> RecordingResult:
        with self._lock:
            if not self._recording:
                raise RecordingError("当前没有正在进行的录音。")
            stream = self._stream
            numpy = self._numpy
            self._recording = False
            self._started_at = 0.0
            self._stream = None

        try:
            if stream is not None:
                stream.stop()
                stream.close()
        except Exception as exc:
            raise RecordingError(f"停止录音失败：{exc}") from exc

        if not self._chunks:
            raise RecordingError("录音内容为空。")

        assert numpy is not None
        try:
            audio_array = numpy.concatenate(self._chunks, axis=0)
        except Exception as exc:
            raise RecordingError(f"拼接录音数据失败：{exc}") from exc

        if len(audio_array) == 0:
            raise RecordingError("录音内容为空。")

        timestamp = utc_now_iso().replace(":", "-")
        file_path = self.temp_dir / f"segment_{timestamp}_{int(time.time() * 1000)}.wav"
        try:
            with wave.open(str(file_path), "wb") as wav_file:
                wav_file.setnchannels(self.channels)
                wav_file.setsampwidth(2)
                wav_file.setframerate(self.sample_rate)
                wav_file.writeframes(audio_array.tobytes())
        except Exception as exc:
            raise RecordingError(f"保存录音文件失败：{exc}") from exc

        duration_sec = round(len(audio_array) / float(self.sample_rate), 2)
        self._chunks = []
        return RecordingResult(audio_path=str(file_path), duration_sec=duration_sec)

    def discard(self) -> None:
        with self._lock:
            stream = self._stream
            self._stream = None
            self._recording = False
            self._started_at = 0.0
            self._chunks = []
        if stream is not None:
            try:
                stream.stop()
                stream.close()
            except Exception:
                pass


def cleanup_audio_file(audio_path: str) -> None:
    try:
        Path(audio_path).unlink(missing_ok=True)
    except Exception:
        pass


class GlobalHotkeyMonitor:
    def __init__(
        self,
        *,
        hotkey: str,
        on_press: Callable[[], None],
        on_release: Callable[[], None],
        poll_interval_sec: float = 0.05,
    ) -> None:
        self.hotkey = hotkey
        self.on_press = on_press
        self.on_release = on_release
        self.poll_interval_sec = poll_interval_sec
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._pressed = False

    def start(self) -> None:
        try:
            import keyboard  # type: ignore
        except ImportError as exc:
            raise HotkeyError(
                "global hotkey 功能缺少依赖，请运行：python -m pip install keyboard"
            ) from exc

        try:
            keyboard.parse_hotkey(self.hotkey)
            keyboard.is_pressed(self.hotkey)
        except Exception as exc:
            raise HotkeyError(f"注册 global hotkey 失败：{exc}") from exc

        def run() -> None:
            try:
                while not self._stop_event.is_set():
                    is_pressed = keyboard.is_pressed(self.hotkey)
                    if is_pressed and not self._pressed:
                        self._pressed = True
                        self.on_press()
                    elif not is_pressed and self._pressed:
                        self._pressed = False
                        self.on_release()
                    time.sleep(self.poll_interval_sec)
            except Exception:
                # Main loop handles fallback and teardown; keep this worker quiet.
                return

        self._thread = threading.Thread(target=run, name="global-hotkey-monitor", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
