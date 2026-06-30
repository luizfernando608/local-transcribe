"""Operações de áudio via ffmpeg/ffprobe."""
from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path


def ensure_ffmpeg() -> None:
    if not shutil.which("ffmpeg") or not shutil.which("ffprobe"):
        raise RuntimeError("ffmpeg/ffprobe não encontrados no PATH.")


def _run(cmd: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, check=True, capture_output=True, text=True)


def duration_seconds(path: str | Path) -> float:
    out = _run([
        "ffprobe", "-v", "error", "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1", str(path),
    ])
    return float(out.stdout.strip())


def extract_wav(src: str | Path, dst: str | Path, sr: int = 16000) -> Path:
    """Extrai/converte para WAV PCM16 mono no sample rate dado (ideal p/ Whisper e pyannote)."""
    ensure_ffmpeg()
    _run([
        "ffmpeg", "-y", "-i", str(src), "-vn", "-ac", "1", "-ar", str(sr),
        "-c:a", "pcm_s16le", str(dst),
    ])
    return Path(dst)


def chunk_to_flac(src_wav: str | Path, dst: str | Path, start: float, length: float, sr: int = 16000) -> Path:
    """Recorta [start, start+length] e grava FLAC 16k mono (pequeno, p/ upload no Groq)."""
    _run([
        "ffmpeg", "-y", "-ss", f"{start:.3f}", "-t", f"{length:.3f}", "-i", str(src_wav),
        "-ac", "1", "-ar", str(sr), str(dst),
    ])
    return Path(dst)
