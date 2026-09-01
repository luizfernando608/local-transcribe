#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""local-transcribe — CLI de transcrição + diarização.

Uso:
    python transcribe.py "entrada.mp4"
    python transcribe.py "audio.m4a" --provider local --model large-v3 --speakers 2
    python transcribe.py "reuniao.mp4" --glossary glossaries/exemplo.txt --out saida/

Chaves (lidas do ambiente ou de .env conhecidos — nunca versionadas):
    GROQ_API_KEY  -> transcrição rápida via Groq (Whisper large-v3)
    HF_TOKEN      -> diarização pyannote (gold standard)
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

# permite rodar sem instalar (src/ no path)
sys.path.insert(0, str(Path(__file__).parent / "src"))

from ltranscribe import audio, config, correct, diarize, merge, output  # noqa: E402


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Transcrição + diarização local (Groq opcional).")
    ap.add_argument("input", help="arquivo de áudio ou vídeo")
    ap.add_argument("--out", help="pasta de saída (padrão: <entrada>_transcricao)")
    ap.add_argument("--provider", choices=["auto", "groq", "local"], default="auto")
    ap.add_argument("--language", default="pt", help="idioma (ISO) ou 'auto'")
    ap.add_argument("--model", default="large-v3", help="modelo faster-whisper local")
    ap.add_argument("--groq-model", default="whisper-large-v3")
    ap.add_argument("--prompt", help="termos/jargões para orientar a transcrição")
    ap.add_argument("--no-diarize", action="store_true", help="pula a diarização")
    ap.add_argument("--diarizer", choices=["auto", "onnx", "pyannote", "resemblyzer"], default="auto")

    ap.add_argument("--diar-model", default="pyannote/speaker-diarization-community-1",
                    help="modelo pyannote (ex.: pyannote/speaker-diarization-3.1)")
    ap.add_argument("--speakers", type=int, help="nº exato de falantes (se souber)")
    ap.add_argument("--min-speakers", type=int)
    ap.add_argument("--max-speakers", type=int)
    ap.add_argument("--glossary", help="arquivo de glossário (regras 'regex => troca')")
    ap.add_argument("--cpu-threads", type=int, default=8)
    ap.add_argument("--keep-wav", action="store_true")
    args = ap.parse_args(argv)

    src = Path(args.input)
    if not src.exists():
        log(f"ERRO: arquivo não encontrado: {src}")
        return 2

    out_dir = Path(args.out) if args.out else src.parent / f"{src.stem}_transcricao"
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = out_dir / src.stem
    lang = None if args.language.lower() == "auto" else args.language.lower()

    loaded = config.load_env_files()
    if loaded:
        log(f"Chaves carregadas de: {', '.join(loaded)}")
    groq_key = config.get_key("GROQ_API_KEY")
    hf_token = config.get_key("HF_TOKEN")
    log(f"Groq: {'disponível' if groq_key else 'ausente'} | HF_TOKEN: {'disponível' if hf_token else 'ausente'}")

    # 1) áudio
    wav = out_dir / f"{src.stem}_16k.wav"
    log(f"Extraindo áudio -> {wav.name}")
    audio.extract_wav(src, wav)
    dur = audio.duration_seconds(wav)
    log(f"Duração: {dur:.0f}s ({dur / 60:.1f} min)")

    # 2) transcrição
    t0 = time.time()
    from ltranscribe import transcribe as tr
    transcript = tr.transcribe(
        wav, provider=args.provider, groq_api_key=groq_key,
        model_local=args.model, groq_model=args.groq_model,
        language=lang, prompt=args.prompt, cpu_threads=args.cpu_threads, log=log,
    )
    log(f"Transcrição: {len(transcript['segments'])} segmentos em {time.time() - t0:.0f}s")
    output.write_transcript(transcript, stem)

    # 3) diarização + merge
    diarized_lines = None
    if not args.no_diarize:
        diarizer = args.diarizer
        if diarizer == "auto":
            diarizer = "onnx"
        log(f"Diarização via {diarizer}...")
        t1 = time.time()
        try:
            if diarizer == "onnx":
                turns = diarize.diarize_onnx(
                    wav, num_speakers=args.speakers, log=log,
                )
            elif diarizer == "pyannote":
                if not hf_token:
                    raise RuntimeError("HF_TOKEN ausente para pyannote.")
                turns = diarize.diarize_pyannote(
                    wav, hf_token, model=args.diar_model, num_speakers=args.speakers,
                    min_speakers=args.min_speakers, max_speakers=args.max_speakers, log=log,
                )
            else:
                turns = diarize.diarize_resemblyzer(
                    wav, transcript["segments"], num_speakers=args.speakers,
                    max_speakers=args.max_speakers or 4, log=log,
                )
            n_spk = len({t["speaker"] for t in turns})
            log(f"Diarização: {len(turns)} turnos, {n_spk} falantes em {time.time() - t1:.0f}s")

            diarized_lines = merge.merge(transcript, turns)
            output.write_diarized(diarized_lines, out_dir / f"{src.stem}_falantes.txt")
            output.write_diarized_json(diarized_lines, out_dir / f"{src.stem}_falantes.json")
        except Exception as exc:  # noqa: BLE001
            log(f"AVISO: diarização falhou ({exc!r}). Seguindo só com a transcrição.")

    # 4) correção por glossário
    if args.glossary:
        base = (out_dir / f"{src.stem}_falantes.txt")
        if not base.exists():
            base = stem.with_suffix(".txt")
        text = base.read_text(encoding="utf-8")
        rules = correct.load_glossary(args.glossary)
        fixed, counts = correct.apply_glossary(text, rules)
        total = sum(counts.values())
        md = out_dir / f"{src.stem}_corrigido.md"
        header = (
            f"# Transcrição revisada (correção de glossário)\n\n"
            f"> Base: `{base.name}` · {total} substituições de alta confiança.\n\n---\n\n"
        )
        md.write_text(header + fixed, encoding="utf-8")
        log(f"Correção de glossário: {total} trocas -> {md.name}")

    if not args.keep_wav:
        try:
            wav.unlink()
        except OSError:
            pass

    log(f"PRONTO. Saídas em: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
