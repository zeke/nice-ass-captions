# AGENTS.md — nice-ass-captions

Instructions for agents working with this repo.

## What this repo does

`caption.py` burns word-highlighted captions into a video. Full pipeline:

1. Extract 16kHz mono WAV (ffmpeg)
2. Transcribe with word-level timestamps (whisper-cli --dtw --output-words)
3. Parse .wts bash script → merge compound words, contractions, punctuation
4. Generate ASS subtitle file with rounded box + per-word alpha animations
5. Burn ASS into video (ffmpeg-full with libass)

## Running the tool

```sh
uv run caption video.mp4
uv run caption video.mp4 --output out.mp4 --words 5 --model large-v3-turbo
uv run caption video.mp4 --prompt "Cloudflare, WebAssembly, MyProductName"
uv run caption video.mp4 --transcript script.txt
uv run caption video.mp4 --colorize global
uv run caption video.mp4 --colorize per-chunk
uv run caption video.mp4 --position top
uv run caption video.mp4 --words-json precomputed-words.json
```

## Model paths

Default model directory: `~/.cache/nice-ass-captions/`

Models on this machine (zeke's Mac):
- `~/.cache/nice-ass-captions/ggml-large-v3-turbo.bin` (1.6GB, recommended/default)
- `~/.cache/nice-ass-captions/ggml-large-v3.bin` (3.1GB, highest accuracy)
- `~/.cache/nice-ass-captions/ggml-medium.en.bin` (1.5GB)
- `~/.cache/nice-ass-captions/ggml-small.en.bin` (466MB)
- `~/.cache/nice-ass-captions/ggml-base.en.bin` (142MB)

The script auto-detects any `ggml-*.bin` file in that directory, preferring `large-v3-turbo`.

Note: whisper.cpp's `--dtw` presets for large models are dotted (`large.v3`, `large.v3.turbo`)
while the filenames are hyphenated; `_model_short_name()` converts `large-*` names accordingly.

## ffmpeg paths

The script checks these in order for an ffmpeg with libass support:
1. `/opt/homebrew/opt/ffmpeg-full/bin/ffmpeg` — Apple Silicon
2. `/usr/local/opt/ffmpeg-full/bin/ffmpeg` — Intel Mac
3. `ffmpeg` from PATH (validated by checking for `subtitles` in `-filters` output)

## Key constants in caption.py

These are the main knobs. Edit them directly:

| Constant | Location | Effect |
| -------- | -------- | ------ |
| `FONT_SIZE` | top of file | Text size — scale with resolution |
| `PAD_X / PAD_Y` | top of file | Box padding — increase for more breathing room |
| `CORNER_R` | top of file | Box corner roundness |
| `ALPHA_DIM` | top of file | Inactive word transparency (`&H99&` ≈ 60% opaque) |
| `COL_TEXT` | top of file | Caption text color |
| `COL_BOX` | top of file | Box color+opacity (ASS `&HAABBGGRR`) |
| `MIN_CONTRAST` | top of file | Minimum contrast ratio for `--colorize` |
| `WORDS_PER_CHUNK` | top of file | Words per line — 4-6 is the sweet spot |
| `CAPTION_POSITION` | top of file | Default caption placement: `top`, `center`, or `bottom` |
| `COMPOUND_MERGES` | top of file | Token sequences to join (see below) |

## Extending COMPOUND_MERGES

whisper.cpp sometimes splits compound words at subword boundaries. Add entries as you find them:

```python
COMPOUND_MERGES = [
    ["cloud", "fl", "are"],   # Cloudflare
    ["b", "rows", "ers"],     # Browsers
    ["my", "product"],        # MyProduct — add your own
]
```

Each entry is a list of lowercase token strings. The merge is case-insensitive and
preserves the original casing of the first token. Punctuation attached to the last
token is also preserved.

## Using --prompt for proper nouns

The `--prompt` flag passes an initial context string to whisper before transcription.
This biases the model toward recognizing specific words:

```sh
uv run caption video.mp4 --prompt "Cloudflare Browser Rendering, don't, you're"
```

Good for: product names, technical terms, names of people. Contractions in the prompt
help the model transcribe them correctly too.

Use `--transcript script.txt` when raw captions or a script are available. The file is
passed to whisper.cpp as an initial prompt with `--carry-initial-prompt`; this can help
spellings and punctuation, but it is not forced alignment and does not guarantee exact
caption text.

## Using --words-json with an external model

`--words-json path.json` bypasses `extract_audio()`/`transcribe()`/`parse_wts()` entirely and
feeds a pre-built `[{word, start, end}, ...]` array (seconds, absolute video timeline, sorted
by `start`) straight into `chunk_words()`. No whisper.cpp, no model file, no ffmpeg audio
extraction. See the README's "Bringing your own word timings" section for when this is the
right call (multilingual/code-switched audio, or when you already have a known-correct
transcript to align against).

This was built and proven out captioning a multilingual (English/Spanish/French) phone-call
recording for the `dial-a-repo` project, using [Replicate](https://replicate.com)'s hosted
models instead of local whisper.cpp. Two real gotchas came out of that, worth knowing before
reaching for a Replicate forced-alignment model again:

**`quinten-kamphuis/forced-alignment` (torchaudio MMS) silently falls back to uniform word
spacing on failure — it does not raise an error.** Its `predict.py` wraps the whole alignment
call in a bare `except Exception`, and on failure divides the clip's total duration evenly
across all words instead. The output still looks like a normal `{word, start, end}` list, so
nothing about the response signals that timing is wrong. Detect it by checking whether every
word has the exact same `end - start` duration:

```python
durs = set(round(w["end"] - w["start"], 4) for w in words)
uniform_fallback = len(durs) == 1 and len(words) > 3
```

Two known triggers for this specific model:

- **Hyphens in the script.** `-` isn't in the model's character dictionary. A word like
  `dial-a-repo` or `veux-tu` reproducibly broke alignment for that whole clip. Fix: replace
  hyphens with spaces in the script text before sending it (`dial a repo`, `veux tu`) — the
  model still returns each piece as a separate timed word, which is fine for captioning.
- **Clips longer than roughly 10–15 seconds of dense speech.** Reproducible on a real 14.5s/25-word
  clip that failed every retry, while the same text split into two ~7s halves each aligned
  correctly. The model doesn't chunk long audio internally, so treat anything past ~10s as at
  risk and split by sentence/utterance boundaries rather than assuming a single call over a
  multi-minute file will work.

**Whisper-family models lock onto one language for the whole file.** Confirmed with local
`whisper-cli --model large-v3 -l auto` on real code-switched audio: it transcribed a Spanish
question ("¿Puedes hablar en español también?") as unrelated English text ("Tell me, can you
speak Spanish as well?") instead of switching languages. This is a fundamental limitation of
how these models decode (language is normally detected once from the first ~30s and reused for
the rest of the file), not specific to whisper.cpp — hosted multilingual Whisper variants
(`whisperx`, `incredibly-fast-whisper`) have the same behavior. If a clip code-switches, either
slice it at the language boundaries and transcribe/align each slice with its own forced
language, or use forced alignment against known-correct multilingual text instead of
transcription (see above).

**Whisper-family word-level timestamp quality varies more at the transcript level than the
timestamp level.** Comparing `victor-upmeet/whisperx` (`align_output=True`) against
`vaibhavs10/incredibly-fast-whisper` (`timestamp="word"`) on the same ~3-minute single-speaker
English narration clip, timestamps from both were comparably accurate, but `whisperx` degraded
to a lowercase, unpunctuated run-on for roughly the back half of the clip while
`incredibly-fast-whisper` kept consistent capitalization and punctuation throughout. Worth
re-comparing on a case-by-case basis rather than assuming one is always better — this was one
data point on one clip.

## Common failure modes

**"No whisper model found"**
Download one: `curl -L -o ~/.cache/nice-ass-captions/ggml-large-v3-turbo.bin https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-large-v3-turbo.bin`

**"No ffmpeg with libass"**
Install: `brew install ffmpeg-full` (the standard `ffmpeg` formula doesn't include libass)

**"0 words found" after parse_wts**
The .wts regex failed to match. Run with `--keep-tmp`, inspect the `.wts` file, and check if whisper output format has changed.

**Word appearing split (e.g. "Cloud fl are")**
Add it to `COMPOUND_MERGES` in `caption.py`. Also try `--prompt` with the correct spelling.

**Apostrophe appearing as separate token ("don ' t")**
The contraction regex handles `'t`, `'s`, `'re`, `'ve`, `'ll`, `'d`, `'m`. If a new pattern appears, extend the `CONTRACTION` regex in `parse_wts()`.

## Video resolution

The script auto-detects video dimensions via ffprobe and passes them to `build_ass()`.
Style constants (`FONT_SIZE`, `PAD_X`, etc.) are tuned for 1920x1080. For other
resolutions, scale proportionally.

## Related

- Research notes: https://github.com/zeke/subtitle-research
- libass: https://github.com/libass/libass
- whisper.cpp: https://github.com/ggml-org/whisper.cpp
- ASS format spec: https://github.com/libass/libass/wiki/ASS-File-Format-Guide
- Aegisub ASS tags: https://aegisub.org/docs/latest/ass_tags/
