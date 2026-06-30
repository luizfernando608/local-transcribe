"""Transcrição: backend Groq (rápido, nuvem) com fallback faster-whisper (local).

Formato unificado de saída:
  {"language": str, "duration": float,
   "segments": [{"start","end","text","words":[{"start","end","word","prob"}]}]}
"""
from __future__ import annotations

import math
import tempfile
from pathlib import Path
from typing import Any, Callable

from . import audio


def _g(obj: Any, key: str, default=None):
    """Acessa campo de objeto do SDK (dict ou pydantic)."""
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


# --------------------------- Groq (nuvem) --------------------------- #

def transcribe_groq(
    wav: str | Path,
    api_key: str,
    *,
    model: str = "whisper-large-v3",
    language: str | None = "pt",
    prompt: str | None = None,
    chunk_seconds: int = 600,
    log: Callable[[str], None] = print,
) -> dict:
    """Transcreve via Groq. Faz chunking por tempo p/ respeitar limite de upload
    e desloca os timestamps de cada pedaço. Pede word+segment timestamps."""
    from groq import Groq

    client = Groq(api_key=api_key)
    total = audio.duration_seconds(wav)
    segments: list[dict] = []
    tmpdir = Path(tempfile.mkdtemp(prefix="ltz_groq_"))

    n_chunks = max(1, math.ceil(total / chunk_seconds))
    for i in range(n_chunks):
        start = i * chunk_seconds
        length = min(chunk_seconds, total - start)
        if length <= 0:
            break
        flac = tmpdir / f"chunk_{i:03d}.flac"
        audio.chunk_to_flac(wav, flac, start, length)
        log(f"  Groq chunk {i + 1}/{n_chunks} ({start:.0f}-{start + length:.0f}s)...")
        req: dict[str, Any] = {
            "model": model,
            "response_format": "verbose_json",
            "timestamp_granularities": ["word", "segment"],
            "temperature": 0.0,
        }
        if language:
            req["language"] = language
        if prompt:
            req["prompt"] = prompt
        with flac.open("rb") as fh:
            resp = client.audio.transcriptions.create(file=fh, **req)

        words_all = _g(resp, "words", []) or []
        for seg in (_g(resp, "segments", []) or []):
            s0 = float(_g(seg, "start", 0.0)) + start
            s1 = float(_g(seg, "end", 0.0)) + start
            seg_words = []
            for w in words_all:
                wm = (float(_g(w, "start", 0.0)) + float(_g(w, "end", 0.0))) / 2 + start
                if s0 <= wm <= s1:
                    seg_words.append({
                        "start": float(_g(w, "start", 0.0)) + start,
                        "end": float(_g(w, "end", 0.0)) + start,
                        "word": str(_g(w, "word", "")),
                        "prob": 1.0,
                    })
            segments.append({
                "start": s0, "end": s1,
                "text": str(_g(seg, "text", "")).strip(),
                "words": seg_words,
            })

    return {"language": language or "auto", "duration": total, "segments": segments}


# --------------------------- faster-whisper (local) --------------------------- #

def transcribe_local(
    wav: str | Path,
    *,
    model_size: str = "large-v3",
    language: str | None = "pt",
    compute_type: str = "int8",
    cpu_threads: int = 8,
    beam_size: int = 5,
    prompt: str | None = None,
    log: Callable[[str], None] = print,
) -> dict:
    from faster_whisper import WhisperModel

    log(f"  Carregando faster-whisper {model_size} ({compute_type}, CPU)...")
    model = WhisperModel(model_size, device="cpu", compute_type=compute_type, cpu_threads=cpu_threads)
    seg_iter, info = model.transcribe(
        str(wav),
        language=language,
        beam_size=beam_size,
        vad_filter=True,
        vad_parameters={"min_silence_duration_ms": 500},
        word_timestamps=True,
        condition_on_previous_text=True,
        initial_prompt=prompt,
    )
    segments: list[dict] = []
    for s in seg_iter:
        words = [
            {"start": w.start, "end": w.end, "word": w.word, "prob": w.probability}
            for w in (s.words or [])
        ]
        segments.append({"start": s.start, "end": s.end, "text": s.text.strip(), "words": words})
        if len(segments) % 20 == 0:
            log(f"    {len(segments)} segs | áudio {s.end:.0f}s/{info.duration:.0f}s")
    return {"language": info.language, "duration": info.duration, "segments": segments}


def transcribe(
    wav: str | Path,
    *,
    provider: str = "auto",
    groq_api_key: str | None = None,
    model_local: str = "large-v3",
    groq_model: str = "whisper-large-v3",
    language: str | None = "pt",
    prompt: str | None = None,
    cpu_threads: int = 8,
    log: Callable[[str], None] = print,
) -> dict:
    """Despacha para Groq ou local conforme `provider` e disponibilidade da chave."""
    use_groq = provider == "groq" or (provider == "auto" and groq_api_key)
    if use_groq:
        if not groq_api_key:
            raise RuntimeError("provider=groq mas GROQ_API_KEY ausente.")
        try:
            log("Transcrição via Groq (whisper-large-v3)...")
            return transcribe_groq(wav, groq_api_key, model=groq_model, language=language, prompt=prompt, log=log)
        except Exception as exc:  # noqa: BLE001
            if provider == "groq":
                raise
            log(f"  Groq falhou ({exc!r}); caindo para local.")
    log("Transcrição local (faster-whisper)...")
    return transcribe_local(wav, model_size=model_local, language=language, cpu_threads=cpu_threads, prompt=prompt, log=log)
