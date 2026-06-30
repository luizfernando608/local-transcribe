"""Correções de glossário de ALTA confiança (siglas/jargões mal transcritos).

Formato do arquivo de glossário (uma regra por linha; '#' = comentário):
    padrao_regex => substituto
Ex.:
    \\bcarro\\b => CAR
    \\bShopify\\b => shapefile

A correção contextual mais profunda (entender qual palavra estava errada pelo
sentido) é feita por um humano/LLM lendo o texto — este passo só aplica trocas
seguras e determinísticas. Veja o README.
"""
from __future__ import annotations

import re
from pathlib import Path


def load_glossary(path: str | Path) -> list[tuple[str, str]]:
    rules: list[tuple[str, str]] = []
    for raw in Path(path).read_text(encoding="utf-8-sig").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=>" not in line:
            continue
        pat, rep = line.split("=>", 1)
        rules.append((pat.strip(), rep.strip()))
    return rules


def apply_glossary(text: str, rules: list[tuple[str, str]]) -> tuple[str, dict[str, int]]:
    counts: dict[str, int] = {}
    for pat, rep in rules:
        text, n = re.subn(pat, rep, text, flags=re.IGNORECASE)
        if n:
            counts[pat] = n
    return text, counts
