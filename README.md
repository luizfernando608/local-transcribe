# local-transcribe

Transcrição + diarização de áudio/vídeo **rodando localmente**, com aceleração
opcional pela API do **Groq** (Whisper large-v3). Pensado para reuniões,
entrevistas e tutoriais em pt-BR, mas funciona para qualquer idioma.

Pipeline:

1. **Áudio** — extrai com `ffmpeg` para WAV 16 kHz mono.
2. **Transcrição** — Groq Whisper `large-v3` se houver `GROQ_API_KEY`; senão
   `faster-whisper` local (CPU, int8). Sempre com timestamps por palavra.
3. **Diarização** — `pyannote/speaker-diarization-3.1` se houver `HF_TOKEN`
   (gold standard); senão fallback leve com **Resemblyzer** + clustering.
4. **Merge** — atribui cada palavra ao falante e agrupa em falas com timestamps.
5. **Correção** — aplica um glossário de trocas de alta confiança (siglas/jargões).

## Requisitos

- Python 3.11+
- `ffmpeg` e `ffprobe` no PATH
- `pip install -r requirements.txt`
- Diarização: `pip install -r requirements-diarize.txt`

## Chaves (opcionais, nunca versionadas)

Defina por ambiente, por `.env` na pasta atual, ou deixe que o programa
reaproveite os `.env` de projetos conhecidos desta máquina
(ver `DEFAULT_ENV_SEARCH` em `src/ltranscribe/config.py`):

| Variável        | Para quê                          | Sem ela                     |
|-----------------|-----------------------------------|-----------------------------|
| `GROQ_API_KEY`  | transcrição rápida (Groq)         | usa faster-whisper local    |
| `HF_TOKEN`      | diarização pyannote (gold)        | usa fallback Resemblyzer    |

> `pyannote/speaker-diarization-3.1` exige aceitar os termos dos modelos
> `pyannote/speaker-diarization-3.1` e `pyannote/segmentation-3.0` no HuggingFace
> com a conta dona do `HF_TOKEN`.

## Uso

```bash
# auto: usa Groq+pyannote se as chaves existirem, senão local
python transcribe.py "reuniao.mp4"

# 100% local, modelo grande, 2 falantes conhecidos
python transcribe.py "audio.m4a" --provider local --model large-v3 --speakers 2

# com glossário de domínio e pasta de saída
python transcribe.py "tutorial.mp4" --glossary glossaries/exemplo-car.txt --out saida/

# orientar a transcrição com jargões
python transcribe.py "call.mp4" --prompt "SICAR, SIRGAS 2000, shapefile, QGIS"
```

### Principais flags

| Flag | Default | Descrição |
|------|---------|-----------|
| `--provider` | `auto` | `auto` / `groq` / `local` |
| `--language` | `pt` | ISO do idioma ou `auto` |
| `--model` | `large-v3` | modelo faster-whisper local |
| `--diarizer` | `auto` | `auto` / `pyannote` / `resemblyzer` |
| `--speakers N` | — | nº exato de falantes (se souber) |
| `--min-speakers`/`--max-speakers` | — | faixa de falantes |
| `--glossary FILE` | — | regras `regex => troca` |
| `--no-diarize` | — | só transcreve |
| `--cpu-threads` | `8` | threads no caminho local |
| `--keep-wav` | — | mantém o WAV extraído |

## Saídas (em `<entrada>_transcricao/`)

- `<nome>.json` — transcrição com timestamps por palavra
- `<nome>.srt` / `<nome>.txt` — legenda e texto corrido
- `<nome>_falantes.txt` / `.json` — transcrição diarizada (com falantes)
- `<nome>_corrigido.md` — transcrição após o glossário (se `--glossary`)

## Correção contextual (o passo "humano/LLM")

O glossário só faz trocas determinísticas e seguras. A correção mais profunda —
entender, pelo sentido, qual palavra foi transcrita errada, mapear quem é cada
`SPEAKER_xx` pelos papéis, normalizar nomes próprios — é feita por um humano ou
por um LLM (ex.: Claude Code) lendo `_falantes.txt`. Veja a skill
`transcribe-diarize` em `~/.claude/skills/`.
