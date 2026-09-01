"""Diarização: pyannote (gold standard, requer HF_TOKEN) com fallback Resemblyzer.

Saída: lista de turnos [{"start","end","speaker"}] ordenada por tempo.
"""
from __future__ import annotations

import wave
from pathlib import Path
from typing import Callable

import numpy as np


def _load_wav_mono(path: str | Path):
    """Lê WAV PCM16 mono em memória (evita o torchcodec quebrado no Windows)."""
    with wave.open(str(path), "rb") as wf:
        sr = wf.getframerate()
        ch = wf.getnchannels()
        sampwidth = wf.getsampwidth()
        raw = wf.readframes(wf.getnframes())
    if sampwidth != 2:
        raise ValueError(f"esperado PCM16, sampwidth={sampwidth}")
    data = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    if ch > 1:
        data = data.reshape(-1, ch).mean(axis=1)
    return data, sr


# --------------------------- pyannote --------------------------- #

def diarize_pyannote(
    wav: str | Path,
    hf_token: str,
    *,
    model: str = "pyannote/speaker-diarization-community-1",
    num_speakers: int | None = None,
    min_speakers: int | None = None,
    max_speakers: int | None = None,
    threads: int = 6,
    log: Callable[[str], None] = print,
) -> list[dict]:
    import torch
    from pyannote.audio import Pipeline

    torch.set_num_threads(threads)
    data, sr = _load_wav_mono(wav)
    waveform = torch.from_numpy(data).unsqueeze(0)  # (1, T)

    log(f"  Carregando pipeline {model}...")
    try:  # pyannote.audio 4.x usa token=; 3.x usa use_auth_token=
        pipeline = Pipeline.from_pretrained(model, token=hf_token)
    except TypeError:
        pipeline = Pipeline.from_pretrained(model, use_auth_token=hf_token)
    pipeline.to(torch.device("cpu"))

    kwargs: dict = {}
    if num_speakers:
        kwargs["num_speakers"] = num_speakers
    else:
        if min_speakers:
            kwargs["min_speakers"] = min_speakers
        if max_speakers:
            kwargs["max_speakers"] = max_speakers

    log("  Diarizando (pode demorar em CPU)...")
    result = pipeline({"waveform": waveform, "sample_rate": sr}, **kwargs)

    ann = getattr(result, "speaker_diarization", result)
    turns = [
        {"start": float(t.start), "end": float(t.end), "speaker": str(spk)}
        for t, _, spk in ann.itertracks(yield_label=True)
    ]
    turns.sort(key=lambda x: x["start"])
    return turns


# --------------------------- ONNX (DirectML / CPU Ultra-Fast) --------------------------- #

def diarize_onnx(
    wav: str | Path,
    *,
    model_name: str = "segmentation-3.0",
    num_speakers: int | None = None,
    threshold: float = 0.5,
    log: Callable[[str], None] = print,
) -> list[dict]:
    """Diarização ultra-rápida via ONNX Runtime com aceleração DirectML/CPU."""
    from pyannote_onnx import PyannoteONNX

    data, sr = _load_wav_mono(wav)
    dur = len(data) / sr

    log(f"  Inicializando modelo ONNX ({model_name})...")
    pyannote = PyannoteONNX(model_name=model_name, show_progress=False)

    log("  Inferência de segmentação ONNX...")
    seg_list = list(pyannote(data))
    seg_arr = np.array(seg_list)

    total_frames = len(seg_arr)
    if total_frames == 0:
        return [{"start": 0.0, "end": dur, "speaker": "SPEAKER_00"}]

    time_step = dur / total_frames
    turns: list[dict] = []
    current_speaker = None
    turn_start = 0.0

    for frame_idx, row in enumerate(seg_arr):
        t = frame_idx * time_step
        best_spk = int(np.argmax(row))
        if num_speakers and best_spk >= num_speakers:
            # limita aos top N se especificado
            sorted_spks = np.argsort(row)[::-1]
            valid = [s for s in sorted_spks if s < num_speakers]
            best_spk = valid[0] if valid else 0

        if row[best_spk] > threshold:
            spk_label = f"SPEAKER_{best_spk:02d}"
        else:
            spk_label = None

        if spk_label != current_speaker:
            if current_speaker is not None:
                turns.append({"start": turn_start, "end": t, "speaker": current_speaker})
            current_speaker = spk_label
            turn_start = t

    if current_speaker is not None:
        turns.append({"start": turn_start, "end": dur, "speaker": current_speaker})

    if not turns:
        turns = [{"start": 0.0, "end": dur, "speaker": "SPEAKER_00"}]

    return turns



# --------------------------- Resemblyzer (fallback) --------------------------- #

def diarize_resemblyzer(
    wav: str | Path,
    segments: list[dict],
    *,
    num_speakers: int | None = None,
    max_speakers: int = 4,
    log: Callable[[str], None] = print,
) -> list[dict]:
    """Diarização leve sem token: embeddings por segmento + clustering aglomerativo.
    Usa os segmentos da transcrição para definir as janelas."""
    import librosa
    from resemblyzer import VoiceEncoder
    from sklearn.cluster import AgglomerativeClustering
    from sklearn.metrics import silhouette_score

    sr = 16000
    y, _ = librosa.load(str(wav), sr=sr, mono=True)
    dur = len(y) / sr
    enc = VoiceEncoder("cpu")

    emb, idx = [], []
    MIN = 1.0
    for i, s in enumerate(segments):
        a, b = float(s["start"]), float(s["end"])
        if b - a < MIN:
            c = (a + b) / 2
            a, b = max(0, c - MIN / 2), min(dur, c + MIN / 2)
        sl = y[int(a * sr):int(b * sr)]
        if len(sl) < int(0.4 * sr):
            continue
        try:
            emb.append(enc.embed_utterance(sl))
            idx.append(i)
        except Exception:  # noqa: BLE001
            continue

    if not emb:
        return [{"start": 0.0, "end": dur, "speaker": "SPEAKER_00"}]

    X = np.vstack(emb)
    if num_speakers:
        labels = AgglomerativeClustering(n_clusters=num_speakers, metric="cosine", linkage="average").fit_predict(X)
    else:
        best_k, best_score, labels = 2, -1.0, None
        for k in range(2, min(max_speakers + 1, len(idx))):
            lab = AgglomerativeClustering(n_clusters=k, metric="cosine", linkage="average").fit_predict(X)
            try:
                sc = silhouette_score(X, lab, metric="cosine")
            except Exception:  # noqa: BLE001
                sc = -1.0
            log(f"    k={k} silhouette={sc:.3f}")
            if sc > best_score:
                best_k, best_score, labels = k, sc, lab
        log(f"  Falantes escolhidos: {best_k} (silhouette={best_score:.3f})")

    mapping: dict[int, str] = {}
    for l in labels:
        if l not in mapping:
            mapping[l] = f"SPEAKER_{len(mapping):02d}"

    # constrói turnos a partir dos segmentos rotulados
    seg_speaker = {i: mapping[l] for i, l in zip(idx, labels)}
    turns: list[dict] = []
    last = "SPEAKER_00"
    for i, s in enumerate(segments):
        spk = seg_speaker.get(i, last)
        last = spk
        if turns and turns[-1]["speaker"] == spk:
            turns[-1]["end"] = float(s["end"])
        else:
            turns.append({"start": float(s["start"]), "end": float(s["end"]), "speaker": spk})
    return turns
