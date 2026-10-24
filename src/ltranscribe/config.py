"""Carregamento de chaves de API a partir do ambiente, do cofre do PACE e de arquivos .env.

IMPORTANTE: este módulo só guarda CAMINHOS de onde ler segredos — nunca os
segredos em si. As chaves nunca são impressas nem versionadas no git.
"""
from __future__ import annotations

import os
from pathlib import Path

HOME = Path.home()

# Locais onde procurar arquivos .env com chaves (GROQ_API_KEY, HF_TOKEN), em ordem.
# Reaproveita os .env de projetos já existentes nesta máquina, para "funcionar de cara".
# O primeiro (.env da pasta atual) é o único lido antes do cofre do PACE.
DEFAULT_ENV_SEARCH: list[Path] = [
    Path.cwd() / ".env",
    HOME / "Documents" / "whisper-dictation-tray" / ".env",   # GROQ_API_KEY (legado do PACE)
    Path(os.environ.get("LOCALAPPDATA", HOME / "AppData" / "Local")) / "Pace" / ".env",  # PACE instalado (versões antigas)
    HOME / "OneDrive" / "Documentos" / "Trabalho com CAR" / ".env",  # HF_TOKEN
    HOME / ".config" / "local-transcribe" / ".env",
]

# O PACE (whisper-dictation-tray) guarda as chaves no Gerenciador de Credenciais do
# Windows via `keyring`, como (serviço, provedor). É lá que a janela de Configurações
# dele grava uma chave nova; por isso o cofre vem antes dos .env compartilhados, que
# podem ter ficado com uma chave antiga.
PACE_KEYRING: dict[str, list[tuple[str, str]]] = {
    "GROQ_API_KEY": [("PACE", "groq"), ("WhisperDictationTray", "groq")],
}


def load_env_files(extra: str | os.PathLike | None = None) -> list[str]:
    """Carrega as chaves sem sobrescrever o que já está no ambiente, nesta ordem:
    `extra`, .env da pasta atual, cofre do PACE, demais .env de DEFAULT_ENV_SEARCH.

    Retorna as origens efetivamente lidas (para log), nunca os valores.
    """
    local = ([Path(extra)] if extra else []) + DEFAULT_ENV_SEARCH[:1]
    return _load_paths(local) + _load_pace_keyring() + _load_paths(DEFAULT_ENV_SEARCH[1:])


def _load_paths(paths: list[Path]) -> list[str]:
    loaded: list[str] = []
    for p in paths:
        try:
            if p and p.exists():
                _load_one(p)
                loaded.append(str(p))
        except OSError:
            continue
    return loaded


def _load_pace_keyring() -> list[str]:
    """Lê as chaves do cofre do PACE. Sem `keyring` (ou fora do Windows), não faz nada."""
    try:
        import keyring
    except ImportError:
        return []
    loaded: list[str] = []
    for env_name, entries in PACE_KEYRING.items():
        if env_name in os.environ:  # ambiente tem prioridade
            continue
        for service, user in entries:
            try:
                value = keyring.get_password(service, user)
            except Exception:  # noqa: BLE001 — cofre indisponível/bloqueado
                value = None
            if value and value.strip():
                os.environ[env_name] = value.strip()
                loaded.append(f"cofre do Windows ({service})")
                break
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
