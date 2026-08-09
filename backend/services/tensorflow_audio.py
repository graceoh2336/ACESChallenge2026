"""Real TensorFlow/YAMNet audio siren-detection service.

Mirrors services/camera.py's design: a background thread owns the audio
source (WAV file or live microphone), runs inference continuously, and
caches the latest AudioReading behind a lock. generate_reading() just
returns that cache, so it's always instant and never blocks the asyncio
broadcast loop in websocket.py on TensorFlow inference.

Why this doesn't reuse src/trainClassifier.py's custom classifier:
The merged TensorFlow branch (src/) trains a small Dense(128->32->1)
classifier on top of YAMNet embeddings (src/trainClassifier.py) and expects
a saved ../models/siren_classifier.keras plus a labelled dataset under
../dataset/sirens and ../dataset/notSirens. None of those exist in this
repo (no models/ directory, no labelled dataset — only one raw demo WAV),
and the training script's GroupShuffleSplit dependency (scikit-learn) isn't
installed. Retraining from scratch isn't reusing the merged code, it's
rebuilding it blind with no data to validate against.

What *is* reusable and complete is src/audioUtils.py's YAMNet loading and
waveform-preprocessing pattern (16kHz mono, ~0.96s minimum window) — kept
here almost verbatim. The one adaptation: YAMNet's own classification head
already scores 521 AudioSet classes per window, six of which are exactly
what this task needs (see _SPECIFIC_CLASS_TO_SIREN_TYPE /
_GENERIC_SIREN_CLASSES below) — "Police car (siren)", "Ambulance (siren)",
"Fire engine, fire truck (siren)", plus generic "Siren" / "Civil defense
siren" / "Emergency vehicle". Using those scores directly, instead of the
missing downstream binary classifier, is the smallest change that (a) only
reuses merged code that's actually complete, and (b) genuinely distinguishes
vehicle types rather than emitting one generic label everywhere.
"""

import csv
import hashlib
import logging
import os
import subprocess
import sys
import threading
import time
import wave
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

# lights.py lives at the repo root — same sys.path patch services/camera.py
# uses, so this module works whether or not camera.py has already run it.
_REPO_ROOT = Path(__file__).resolve().parents[2]
_BACKEND_DIR = Path(__file__).resolve().parents[1]

if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import lights  # noqa: E402  (sys.path must be patched before this import)

from config import settings  # noqa: E402
from models import AudioReading, SirenType  # noqa: E402
from services.audio import AudioDetectionService  # noqa: E402

# Persist the downloaded YAMNet model across restarts instead of re-fetching
# it from tfhub.dev into a throwaway OS temp dir every process start. Must be
# set before tensorflow_hub is imported.
_TFHUB_CACHE_DIR = _BACKEND_DIR / ".tfhub_cache"
os.environ.setdefault("TFHUB_CACHE_DIR", str(_TFHUB_CACHE_DIR))

import tensorflow_hub as hub  # noqa: E402

logger = logging.getLogger(__name__)

_YAMNET_URL = "https://tfhub.dev/google/yamnet/1"

# YAMNet needs ~0.96s of audio to produce one embedding/score frame (see
# src/audioUtils.py's MIN_SAMPLES) — these windows stay comfortably above
# that with margin.
_FILE_WINDOW_SECONDS = 1.0
_LIVE_WINDOW_SECONDS = 1.0
_LIVE_HOP_SECONDS = 0.25

_LIVE_SOURCE_KEYWORDS = {"mic", "live", "microphone"}

# AUDIO_SOURCE values that mean "derive the audio from CAMERA_SOURCE's video
# file" instead of pointing at a separate audio file — the default, so the
# demo video's own baked-in siren audio is the single source of truth for
# both OpenCV and YAMNet (see _extract_audio_from_video below).
_VIDEO_DERIVED_KEYWORDS = {"video", "camera", "auto", "demo", ""}

# Extracted-audio cache: gitignored (backend/.cache/), keyed on the source
# video's name/mtime/size so a changed video.mp4 re-extracts automatically
# but repeated process restarts against the same file don't.
_AUDIO_EXTRACT_CACHE_DIR = _BACKEND_DIR / ".cache" / "audio_extract"

# AudioSet classes YAMNet was trained on that map onto a specific emergency
# vehicle type.
_SPECIFIC_CLASS_TO_SIREN_TYPE = {
    "Police car (siren)": SirenType.POLICE,
    "Ambulance (siren)": SirenType.AMBULANCE,
    "Fire engine, fire truck (siren)": SirenType.FIRE_TRUCK,
}

# AudioSet classes that confirm "a siren" without saying which vehicle.
_GENERIC_SIREN_CLASSES = ("Siren", "Civil defense siren", "Emergency vehicle")


def _clamp01(value: float) -> float:
    return round(min(1.0, max(0.0, value)), 3)


class YamnetSirenClassifier:
    """Loads YAMNet exactly once and scores waveform windows against it."""

    def __init__(self, confidence_threshold: float):
        logger.info("Loading YAMNet model (%s)...", _YAMNET_URL)
        t0 = time.monotonic()
        self._model = hub.load(_YAMNET_URL)

        class_map_path = self._model.class_map_path().numpy().decode("utf-8")
        with open(class_map_path, newline="") as f:
            reader = csv.reader(f)
            next(reader)  # header row
            class_names = [row[2] for row in reader]
        self._class_index = {name: i for i, name in enumerate(class_names)}

        self.confidence_threshold = confidence_threshold
        logger.info(
            "YAMNet model loaded in %.1fs (%d classes)",
            time.monotonic() - t0,
            len(class_names),
        )

    def classify(self, waveform: np.ndarray) -> AudioReading:
        scores, _embeddings, _spectrogram = self._model(waveform)
        mean_scores = scores.numpy().mean(axis=0)

        specific_name = max(
            _SPECIFIC_CLASS_TO_SIREN_TYPE, key=lambda name: mean_scores[self._class_index[name]]
        )
        specific_score = float(mean_scores[self._class_index[specific_name]])

        generic_name = max(_GENERIC_SIREN_CLASSES, key=lambda name: mean_scores[self._class_index[name]])
        generic_score = float(mean_scores[self._class_index[generic_name]])

        # Prefer a specific vehicle-type match over the generic siren classes
        # whenever it clears the threshold, so "Police car (siren)" beats a
        # simultaneously-firing "Siren" class instead of being averaged away.
        if specific_score >= self.confidence_threshold:
            return AudioReading(
                audioDetected=True,
                audioConfidence=_clamp01(specific_score),
                sirenType=_SPECIFIC_CLASS_TO_SIREN_TYPE[specific_name],
            )

        if generic_score >= self.confidence_threshold:
            return AudioReading(
                audioDetected=True,
                audioConfidence=_clamp01(generic_score),
                sirenType=SirenType.SIREN,
            )

        return AudioReading(
            audioDetected=False,
            audioConfidence=_clamp01(max(specific_score, generic_score)),
            sirenType=SirenType.NONE,
        )


_shared_classifier_lock = threading.Lock()
_shared_classifier: Optional[YamnetSirenClassifier] = None


def _get_shared_classifier(confidence_threshold: float) -> YamnetSirenClassifier:
    """Process-wide YAMNet singleton — loaded once, reused by every caller."""
    global _shared_classifier
    with _shared_classifier_lock:
        if _shared_classifier is None:
            _shared_classifier = YamnetSirenClassifier(confidence_threshold)
        else:
            _shared_classifier.confidence_threshold = confidence_threshold
        return _shared_classifier


def _read_wav_mono(path: Path) -> Tuple[np.ndarray, int]:
    """Reads a 16-bit PCM WAV via the stdlib (no librosa/soundfile in this
    venv) and downmixes to mono float32 in [-1, 1]."""
    with wave.open(str(path), "rb") as wf:
        n_channels = wf.getnchannels()
        sampwidth = wf.getsampwidth()
        framerate = wf.getframerate()
        raw = wf.readframes(wf.getnframes())

    if sampwidth != 2:
        raise ValueError(
            f"Unsupported WAV sample width ({sampwidth * 8}-bit) in {path}; "
            "only 16-bit PCM is supported."
        )

    data = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    if n_channels > 1:
        data = data.reshape(-1, n_channels).mean(axis=1)
    return data, framerate


def _resample_linear(data: np.ndarray, orig_rate: int, target_rate: int) -> np.ndarray:
    """Simple linear-interpolation resample — scipy/librosa aren't available
    in this venv, and a classification model this robust doesn't need a
    higher-quality resampler for a demo/inference path."""
    if orig_rate == target_rate or len(data) == 0:
        return data
    duration = len(data) / orig_rate
    target_len = max(1, int(round(duration * target_rate)))
    x_old = np.linspace(0, duration, num=len(data), endpoint=False)
    x_new = np.linspace(0, duration, num=target_len, endpoint=False)
    return np.interp(x_new, x_old, data).astype(np.float32)


def _load_waveform_for_yamnet(path: Path, sample_rate: int) -> np.ndarray:
    data, orig_rate = _read_wav_mono(path)
    data = _resample_linear(data, orig_rate, sample_rate)

    min_samples = int(0.98 * sample_rate)
    if len(data) < min_samples:
        data = np.pad(data, (0, min_samples - len(data)))
    return data


def _resolve_audio_path(raw: str) -> Optional[Path]:
    if not raw:
        return None
    candidate = Path(raw)
    if candidate.is_absolute():
        return candidate if candidate.is_file() else None
    for base in (_REPO_ROOT, _BACKEND_DIR):
        resolved = base / raw
        if resolved.is_file():
            return resolved
    return None


def _resolve_video_path(raw_camera_source: str) -> Optional[Path]:
    """Resolves CAMERA_SOURCE to a file path the way services/camera.py's
    own resolver does, except a numeric webcam index has no extractable
    audio track and resolves to None here (live-mic mode should be used for
    that case instead)."""
    raw = raw_camera_source.strip()
    if raw.isdigit():
        return None
    candidate = Path(raw)
    if candidate.is_absolute():
        return candidate if candidate.is_file() else None
    for base in (_REPO_ROOT, _BACKEND_DIR):
        resolved = base / raw
        if resolved.is_file():
            return resolved
    return None


def _extract_audio_from_video(video_path: Path, sample_rate: int) -> Optional[Path]:
    """Extracts the video's audio track to a 16kHz mono WAV once, via a
    bundled ffmpeg binary (imageio-ffmpeg — no system ffmpeg install
    required), and caches it so subsequent process starts against the same
    file reuse it instead of re-decoding the MP4."""
    try:
        stat = video_path.stat()
        cache_key = f"{video_path.resolve()}|{stat.st_mtime_ns}|{stat.st_size}|{sample_rate}"
        digest = hashlib.sha1(cache_key.encode("utf-8")).hexdigest()[:16]
        output_path = _AUDIO_EXTRACT_CACHE_DIR / f"{video_path.stem}_{digest}.wav"

        if output_path.is_file():
            logger.info("Reusing cached extracted audio: %s", output_path)
            return output_path

        import imageio_ffmpeg

        ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()

        _AUDIO_EXTRACT_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        # Drop stale extractions of this same video (old hash) so the cache
        # directory doesn't grow unbounded across repeated edits of video.mp4.
        for stale in _AUDIO_EXTRACT_CACHE_DIR.glob(f"{video_path.stem}_*.wav"):
            stale.unlink(missing_ok=True)

        logger.info("Extracting audio track from %s (once, cached at %s)...", video_path, output_path)
        result = subprocess.run(
            [
                ffmpeg_exe,
                "-y",
                "-i",
                str(video_path),
                "-vn",
                "-ac",
                "1",
                "-ar",
                str(sample_rate),
                "-f",
                "wav",
                str(output_path),
            ],
            capture_output=True,
            text=True,
            timeout=120,
        )
        if result.returncode != 0:
            logger.error(
                "ffmpeg audio extraction failed (code=%d): %s", result.returncode, result.stderr[-2000:]
            )
            output_path.unlink(missing_ok=True)
            return None

        return output_path
    except Exception:
        logger.exception("Failed to extract audio track from %s; audio detection disabled", video_path)
        return None


class TensorFlowAudioDetectionService:
    """Runs YAMNet continuously against a WAV file (looped) or live
    microphone input on a background thread; hands out the latest
    AudioReading on demand. Named generate_reading() to match the interface
    services/audio.py's simulated service already exposes, so
    services/fusion.py needs no changes.
    """

    def __init__(
        self,
        source: Optional[str] = None,
        sample_rate: Optional[int] = None,
        confidence_threshold: Optional[float] = None,
    ):
        self._source = source if source is not None else settings.audio_source
        self._sample_rate = sample_rate if sample_rate is not None else settings.audio_sample_rate
        self._confidence_threshold = (
            confidence_threshold if confidence_threshold is not None else settings.audio_confidence_threshold
        )

        self._classifier: Optional[YamnetSirenClassifier] = None
        self._latest_reading = AudioReading(
            audioDetected=False, audioConfidence=0.0, sirenType=SirenType.NONE
        )

        # Only used if the real model/source can't start AND
        # AUDIO_ALLOW_SIMULATION_FALLBACK=true — see _maybe_start_fallback().
        self._fallback_service: Optional[AudioDetectionService] = None
        self._using_fallback = False

        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        # Set by restart() (see routes/demo.py's POST /api/demo/start) to
        # request that the run loop reset playback to the beginning on its
        # next iteration — used to resync YAMNet's position with a fresh
        # browser video playback and services/camera.py's own restart().
        self._seek_to_start_event = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def start(self) -> None:
        """Loads YAMNet once and starts the continuous inference loop.

        Never raises: a model that can't load or a source that can't open
        leaves the service running with no live detections (or, if
        explicitly configured, simulated ones) rather than taking down the
        FastAPI process it's part of.
        """
        if self._thread is not None or self._using_fallback:
            return

        try:
            self._classifier = _get_shared_classifier(self._confidence_threshold)
        except Exception:
            logger.exception("Failed to load YAMNet model; real audio detection disabled")
            self._maybe_start_fallback()
            return

        mode, target = self._resolve_mode()
        if mode is None:
            logger.warning(
                "Audio detection disabled — no working audio source (AUDIO_SOURCE=%r, "
                "CAMERA_SOURCE=%r). AUDIO_SOURCE='video' (the default) extracts audio from "
                "CAMERA_SOURCE's file; set AUDIO_SOURCE to a WAV file path directly, or 'mic' "
                "for a live microphone.",
                self._source,
                settings.camera_source,
            )
            self._maybe_start_fallback()
            return

        self._stop_event.clear()
        if mode == "file":
            self._thread = threading.Thread(
                target=self._run_file_mode, args=(target,), name="audio-detection-loop", daemon=True
            )
        else:
            self._thread = threading.Thread(
                target=self._run_live_mode, name="audio-detection-loop", daemon=True
            )
        self._thread.start()
        logger.info("Audio detection loop started (mode=%s, source=%r)", mode, self._source)

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=5)
            self._thread = None
        logger.info("Audio detection stopped")

    def restart(self) -> None:
        """Requests the run loop reset playback to the beginning of the
        audio (file mode: back to sample 0; live mode: clear the rolling
        buffer) on its next iteration. No-op if using the simulated
        fallback, which has no playback position to reset."""
        self._seek_to_start_event.set()

    def generate_reading(self) -> AudioReading:
        if self._using_fallback and self._fallback_service is not None:
            return self._fallback_service.generate_reading()
        with self._lock:
            return self._latest_reading

    def _resolve_mode(self) -> Tuple[Optional[str], Optional[Path]]:
        raw = self._source.strip()
        lowered = raw.lower()

        if lowered in _LIVE_SOURCE_KEYWORDS:
            return "live", None

        if lowered in _VIDEO_DERIVED_KEYWORDS:
            video_path = _resolve_video_path(settings.camera_source)
            if video_path is None:
                return None, None
            extracted = _extract_audio_from_video(video_path, self._sample_rate)
            return ("file", extracted) if extracted is not None else (None, None)

        path = _resolve_audio_path(raw)
        if path is not None:
            return "file", path

        return None, None

    def _maybe_start_fallback(self) -> None:
        if not settings.audio_allow_simulation_fallback:
            return
        logger.warning(
            "AUDIO_ALLOW_SIMULATION_FALLBACK is enabled — using simulated audio readings instead."
        )
        self._fallback_service = AudioDetectionService(settings.audio_detection_probability)
        self._using_fallback = True

    def _update_reading(self, reading: AudioReading) -> None:
        with self._lock:
            self._latest_reading = reading
        # Real detection only — wires the genuine detection boolean into
        # lights.py's visual-sensitivity boost. services/camera.py forces
        # this False at import as a safe default and deliberately never lets
        # the *simulated* audio service touch it (see its own comment on
        # this same knob); this is the one place that's allowed to turn it
        # on, because it's now backed by an actual model, not a coin flip.
        lights.AUDIO_SIREN_DETECTED = reading.audioDetected

    def _run_file_mode(self, path: Path) -> None:
        try:
            waveform = _load_waveform_for_yamnet(path, self._sample_rate)
        except Exception:
            logger.exception("Failed to load audio file %s; audio detection disabled", path)
            return

        window_samples = min(int(_FILE_WINDOW_SECONDS * self._sample_rate), len(waveform))
        total_samples = len(waveform)
        position = 0
        next_due = time.monotonic()

        while not self._stop_event.is_set():
            if self._seek_to_start_event.is_set():
                position = 0
                next_due = time.monotonic()
                self._seek_to_start_event.clear()
                logger.info("Audio playback position reset to start")

            end = position + window_samples
            if end <= total_samples:
                chunk = waveform[position:end]
                position = end if end < total_samples else 0
            else:
                # Loop the WAV indefinitely without reloading it — wrap the
                # window across the end/start boundary.
                chunk = np.concatenate([waveform[position:], waveform[: end - total_samples]])
                position = end - total_samples

            try:
                reading = self._classifier.classify(chunk)
                self._update_reading(reading)
            except Exception:
                logger.exception("YAMNet inference failed on a window; skipping")

            # Pace to roughly real-time so the demo video and its extracted
            # audio, started together, stay reasonably in sync — not
            # frame-accurate, just not racing ahead at CPU speed. Waiting on
            # the seek event (rather than time.sleep) means a restart()
            # during this pause is picked up immediately instead of at most
            # one window late.
            next_due += _FILE_WINDOW_SECONDS
            sleep_for = next_due - time.monotonic()
            if sleep_for > 0:
                self._seek_to_start_event.wait(timeout=sleep_for)
            else:
                next_due = time.monotonic()

    def _run_live_mode(self) -> None:
        try:
            import sounddevice as sd
        except Exception as exc:
            logger.warning(
                "Live microphone mode unavailable (%s: %s) — install sounddevice/portaudio "
                "for Raspberry Pi deployment. audioDetected will stay false.",
                type(exc).__name__,
                exc,
            )
            return

        window_samples = int(_LIVE_WINDOW_SECONDS * self._sample_rate)
        hop_samples = int(_LIVE_HOP_SECONDS * self._sample_rate)
        buffer = np.zeros(window_samples, dtype=np.float32)

        try:
            while not self._stop_event.is_set():
                if self._seek_to_start_event.is_set():
                    buffer = np.zeros(window_samples, dtype=np.float32)
                    self._seek_to_start_event.clear()
                    logger.info("Audio playback buffer reset for new demo session")

                chunk = sd.rec(hop_samples, samplerate=self._sample_rate, channels=1, dtype="float32")
                sd.wait()
                buffer = np.concatenate([buffer[hop_samples:], chunk.flatten()])

                try:
                    reading = self._classifier.classify(buffer)
                    self._update_reading(reading)
                except Exception:
                    logger.exception("YAMNet inference failed on a live window; skipping")
        except Exception:
            logger.exception("Live microphone capture failed — audioDetected will stay false.")
