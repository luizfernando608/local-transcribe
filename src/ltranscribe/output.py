"""Escrita de saídas: .json, .srt, .txt e _falantes.txt."""
from __future__ import annotations

import json
from pathlib import Path


def _ts_srt(t: float) -> str:
    h = int(t // 3600); m = int((t % 3600) // 60); s = t % 60
    return f"{h:02d}:{m:02d}:{s:06.3f}".replace(".", ",")


def _ts_hms(t: float) -> str:
    h = int(t // 3600); m = int((t % 3600) // 60); s = int(t % 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def write_transcript(transcript: dict, stem: Path) -> None:
    stem.with_suffix(".json").write_text(
        json.dumps(transcript, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    with stem.with_suffix(".srt").open("w", encoding="utf-8") as srt, \
         stem.with_suffix(".txt").open("w", encoding="utf-8") as txt:
        for i, seg in enumerate(transcript["segments"], 1):
            srt.write(f"{i}\n{_ts_srt(seg['start'])} --> {_ts_srt(seg['end'])}\n{seg['text']}\n\n")
            txt.write(seg["text"] + "\n")


def write_diarized(lines: list[dict], path: Path) -> None:
    with path.open("w", encoding="utf-8") as f:
        for ln in lines:
            f.write(f"[{_ts_hms(ln['start'])} - {_ts_hms(ln['end'])}] {ln['speaker']}: {ln['text']}\n\n")


def write_diarized_json(lines: list[dict], path: Path) -> None:
    path.write_text(json.dumps(lines, ensure_ascii=False, indent=2), encoding="utf-8")
