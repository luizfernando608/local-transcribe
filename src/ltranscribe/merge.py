"""Mescla transcrição (palavra a palavra) com diarização (turnos de falante).

Atribui cada palavra ao falante cujo turno contém o ponto médio da palavra;
agrupa palavras consecutivas do mesmo falante em "falas".
"""
from __future__ import annotations


def _speaker_at(turns: list[dict], t: float) -> str:
    for d in turns:
        if d["start"] <= t <= d["end"]:
            return d["speaker"]
    if not turns:
        return "SPEAKER_00"
    best = min(turns, key=lambda d: min(abs(d["start"] - t), abs(d["end"] - t)))
    return best["speaker"]


def merge(transcript: dict, turns: list[dict]) -> list[dict]:
    turns = sorted(turns, key=lambda d: d["start"])

    words: list[dict] = []
    for seg in transcript["segments"]:
        if seg.get("words"):
            for w in seg["words"]:
                words.append({"start": w["start"], "end": w["end"], "word": w["word"]})
        else:  # segmento sem palavras: usa o texto inteiro
            words.append({"start": seg["start"], "end": seg["end"], "word": seg["text"]})

    for w in words:
        w["speaker"] = _speaker_at(turns, (w["start"] + w["end"]) / 2.0)

    lines: list[dict] = []
    cur: dict | None = None
    for w in words:
        if cur is None or w["speaker"] != cur["speaker"]:
            if cur:
                lines.append(cur)
            cur = {"speaker": w["speaker"], "start": w["start"], "end": w["end"], "_w": [w["word"]]}
        else:
            cur["_w"].append(w["word"])
            cur["end"] = w["end"]
    if cur:
        lines.append(cur)

    import re
    for ln in lines:
        text = " ".join(w.strip() for w in ln.pop("_w") if w.strip())
        text = re.sub(r"\s+([,.;:!?])", r"\1", text)  # remove espaço antes de pontuação
        ln["text"] = text
    return lines
