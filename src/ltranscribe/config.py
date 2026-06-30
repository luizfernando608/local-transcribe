"""Carregamento de chaves de API a partir do ambiente e de arquivos .env.

IMPORTANTE: este módulo só guarda CAMINHOS de onde ler segredos — nunca os
segredos em si. As chaves nunca são impressas nem versionadas no git.
"""
from __future__ import annotations

import os
from pathlib import Path

HOME = Path.home()

# Locais onde procurar arquivos .env com chaves (GROQ_API_KEY, HF_TOKEN), em ordem.
# Reaproveita os .env de projetos já existentes nesta máquina, para "funcionar de cara".
DEFAULT_ENV_SEARCH: list[Path] = [
    Path.cwd() / ".env",
    HOME / "Documents" / "whisper-dictation-tray" / ".env",   # GROQ_API_KEY
    HOME / "OneDrive" / "Documentos" / "Trabalho com CAR" / ".env",  # HF_TOKEN
    HOME / ".config" / "local-transcribe" / ".env",
]


def load_env_files(extra: str | os.PathLike | None = None) -> list[str]:
    """Carrega variáveis dos .env encontrados (sem sobrescrever o que já está no ambiente).

    Retorna a lista de caminhos efetivamente lidos (para log), nunca os valores.
    """
    paths = list(DEFAULT_ENV_SEARCH)
    if extra:
        paths.insert(0, Path(extra))
    loaded: list[str] = []
    for p in paths:
        try:
            if p and p.exists():
                _load_one(p)
                loaded.append(str(p))
        except OSError:
            continue
    return loaded


def _load_one(path: Path) -> None:
    for raw in path.read_text(encoding="utf-8-sig").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        if key.startswith("export "):
            key = key[len("export "):].strip()
        if not key or key in os.environ:  # ambiente tem prioridade
            continue
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]
        os.environ[key] = value


def get_key(name: str) -> str | None:
    value = os.environ.get(name)
    return value.strip() if value else None
