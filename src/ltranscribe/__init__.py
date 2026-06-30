"""local-transcribe: transcrição + diarização local (com aceleração opcional via Groq).

Pipeline:
  1. extrai áudio (ffmpeg) -> WAV 16k mono
  2. transcreve (Groq Whisper large-v3 se houver chave; senão faster-whisper local)
  3. diariza (pyannote se houver HF_TOKEN; senão clustering com Resemblyzer)
  4. mescla palavras + falantes
  5. aplica correções de glossário (alta confiança)
  6. escreve saídas (.json, .srt, .txt, _falantes.txt, _corrigido.md)
"""

__version__ = "0.1.0"
