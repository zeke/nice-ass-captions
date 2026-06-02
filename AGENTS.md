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
uv run caption video.mp4 --output out.mp4 --words 5 --model medium.en
uv run caption video.mp4 --prompt "Cloudflare, WebAssembly, MyProductName"
uv run caption video.mp4 --transcript script.txt
uv run caption video.mp4 --palette-colors
uv run caption video.mp4 --position top
```

## Model paths

Default model directory: `~/.cache/nice-ass-captions/`

Models on this machine (zeke's Mac):
- `~/.cache/nice-ass-captions/ggml-medium.en.bin` (1.5GB, recommended)
- `~/.cache/nice-ass-captions/ggml-small.en.bin` (466MB)
- `~/.cache/nice-ass-captions/ggml-base.en.bin` (142MB)

The script auto-detects any `ggml-*.bin` file in that directory, preferring `medium.en`.

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
| `MIN_CONTRAST` | top of file | Minimum contrast ratio for `--palette-colors` |
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

## Common failure modes

**"No whisper model found"**
Download one: `curl -L -o ~/.cache/nice-ass-captions/ggml-medium.en.bin https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-medium.en.bin`

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
