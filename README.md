# nice-ass-captions

Burn stylish captions into videos using [whisper](https://github.com/ggml-org/whisper.cpp) and [ffmpeg](https://ffmpeg.org). Runs locally. Powered by [ASS™](#the-ass-format).

![screenshot](screenshot.jpg)

## Features

- Word-level highlighting — the active word is bright, the rest are dim
- Rounded semi-transparent background box
- Optional palette-derived text and box colors with accessible contrast
- Accurate local transcription via [whisper.cpp](https://github.com/ggml-org/whisper.cpp) running on-device (Apple Silicon Metal acceleration)
- No API keys, no cloud, no data leaves your machine
- One command from raw video to captioned video

## How it works

1. **Extract audio** — [ffmpeg](https://ffmpeg.org) pulls a 16kHz mono WAV from the input video
2. **Transcribe** — [whisper-cli](https://github.com/ggml-org/whisper.cpp) transcribes the audio with word-level timestamps via DTW forced alignment, writing a `.wts` script with per-word timing windows
3. **Parse and normalize** — the `.wts` is parsed to extract word timestamps; contractions, punctuation, and compound proper nouns (e.g. "Cloudflare") are merged into single tokens
4. **Generate ASS** — words are grouped into ~5-word chunks; each chunk becomes one [ASS](#the-ass-format) `Dialogue` line using `\1a` alpha animations and zero-duration `\t()` transitions to highlight the active word; the background is a `\p1` vector-drawn rounded rectangle
5. **Burn in** — [ffmpeg](https://ffmpeg.org) with [libass](https://github.com/libass/libass) renders the ASS onto the video frames via the `subtitles=` video filter

## The ASS format

[Advanced SubStation Alpha](https://github.com/libass/libass/wiki/ASS-File-Format-Guide) (ASS) is an open subtitle format that has been the standard for high-quality fansubbing since 2003. It is far more expressive than SRT — supporting per-word timing, animations, transparency, rotation, blur, and vector drawing.

[libass](https://github.com/libass/libass) is the open source rendering library for ASS. It is built into [ffmpeg](https://ffmpeg.org), [mpv](https://mpv.io), [VLC](https://www.videolan.org), and most serious media players. nice-ass-captions uses libass via ffmpeg's `subtitles=` video filter to burn captions directly into the video frame.

The specific ASS features this tool uses:

- [`\p1`](https://aegisub.org/docs/latest/ass_tags/#drawing-commands) — vector drawing commands for the rounded rectangle background
- [`\1a`](https://aegisub.org/docs/latest/ass_tags/#set-alpha) — per-word primary alpha (transparency) control
- [`\t(t1,t2,tags)`](https://aegisub.org/docs/latest/ass_tags/#animated-transform) — zero-duration transitions to snap opacity at precise millisecond offsets
- [`\an2\pos(x,y)`](https://aegisub.org/docs/latest/ass_tags/#set-position) — absolute bottom-center positioning so all lines in a chunk render at the same location

The [Aegisub](https://aegisub.org) editor and its [ASS tag reference](https://aegisub.org/docs/latest/ass_tags/) are the best resources for understanding what ASS can do.

## Prerequisites

```sh
brew install whisper-cpp ffmpeg-full
```

[whisper-cpp](https://formulae.brew.sh/formula/whisper-cpp) provides `whisper-cli`. [ffmpeg-full](https://formulae.brew.sh/formula/ffmpeg-full) is the Homebrew variant that includes [libass](https://github.com/libass/libass) support for subtitle rendering. The standard `ffmpeg` formula does not include libass.

## Usage

```sh
# Clone and run
git clone https://github.com/zeke/nice-ass-captions
cd nice-ass-captions
uv run caption video.mp4
uv run caption video.mp4 --position top
```

Output is saved to `video-captioned.mp4` in the same directory as the input.

Use `--palette-colors` to pull global text and background colors from the video:

```sh
uv run caption video.mp4 --palette-colors
```

The palette mode samples frames across the whole video, picks one global color pair,
and checks the text color against the semi-transparent background box for accessible
contrast. If the sampled palette cannot produce a readable pair, it falls back to a
safer high-contrast pair.

Use `--prompt` to give whisper.cpp spelling and vocabulary hints before transcription:

```sh
uv run caption video.mp4 --prompt "Cloudflare, Browser Rendering, WebAssembly, Zeke"
uv run caption video.mp4 --transcript script.txt
```

Good prompt terms include:

- Product names, company names, project names, and people names
- Acronyms, technical terms, jargon, and uncommon words
- Words with expected casing or punctuation, such as `Workers AI`, `don't`, or `you're`

Keep the prompt short and comma-separated. It is context, not a script. The prompt
biases transcription toward those terms, but it does not force them into the output.
Use `--transcript` when you have the raw script or captions. The file is passed to
whisper.cpp as an initial prompt with `--carry-initial-prompt`, so it can improve
spellings and punctuation across segments. It is still guidance, not forced alignment.

## Options

| Flag | Default | Description |
| ---- | ------- | ----------- |
| `-o, --output PATH` | `<input>-captioned.mp4` | Output file path |
| `--model PATH\|NAME` | `~/.cache/nice-ass-captions/ggml-medium.en.bin` | Whisper model path or short name (e.g. `small.en`) |
| `--prompt TEXT` | — | Initial prompt for whisper — improves accuracy for domain-specific proper nouns |
| `--transcript PATH` | — | Raw transcript text file to use as a whisper prompt |
| `--words N` | `5` | Words per caption chunk |
| `--position top\|center\|bottom` | `bottom` | Caption position |
| `--palette-colors` | off | Derive global text and background colors from the video palette |
| `--keep-tmp` | off | Keep intermediate `.wav` and `.wts` files |

## Models

Models live in `~/.cache/nice-ass-captions/`. Download with:

```sh
mkdir -p ~/.cache/nice-ass-captions

# medium.en — recommended (best accuracy/speed tradeoff)
curl -L -o ~/.cache/nice-ass-captions/ggml-medium.en.bin \
  https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-medium.en.bin

# small.en — faster, slightly less accurate
curl -L -o ~/.cache/nice-ass-captions/ggml-small.en.bin \
  https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-small.en.bin
```

| Model | Size | Notes |
| ----- | ---- | ----- |
| `tiny.en` | 75MB | Fastest, basic accuracy |
| `base.en` | 142MB | Fast, decent accuracy |
| `small.en` | 466MB | Good balance |
| `medium.en` | 1.5GB | Best accuracy, still fast on Apple Silicon |

On Apple Silicon, whisper.cpp runs via [Metal](https://developer.apple.com/metal/) (GPU) — `medium.en` transcribes a 2-minute video in under 10 seconds.

## Style

All visual parameters are constants at the top of `caption.py`:

| Constant | Default | Description |
| -------- | ------- | ----------- |
| `FONT_NAME` | `Arial` | Caption font family |
| `FONT_SIZE` | `72` | Font size in script pixels |
| `PAD_X` | `24` | Horizontal padding inside background box |
| `PAD_Y` | `28` | Vertical padding inside background box |
| `CORNER_R` | `18` | Background box corner radius |
| `MARGIN_BOTTOM` | `80` | Distance from bottom of frame to text |
| `CAPTION_POSITION` | `bottom` | Default caption position: `top`, `center`, or `bottom` |
| `ALPHA_DIM` | `&H99&` | Inactive word opacity (~60% opaque) |
| `ALPHA_BRIGHT` | `&H00&` | Active word opacity (fully opaque) |
| `COL_TEXT` | `&H00FFFFFF` | Caption text color |
| `COL_BOX` | `&H48000000` | Background box color and opacity |
| `MIN_CONTRAST` | `4.5` | Minimum contrast ratio for palette colors |
| `WORDS_PER_CHUNK` | `5` | Words per caption group |

Colors use ASS format: `&HAABBGGRR` where `AA` is alpha (`00` = opaque, `FF` = transparent).

## Compound word merges

whisper.cpp occasionally splits compound words at subword boundaries (e.g. "Cloudflare" → `Cloud` + `fl` + `are`). The `COMPOUND_MERGES` list in `caption.py` handles known cases. Add entries for proper nouns specific to your content:

```python
COMPOUND_MERGES = [
    ["cloud", "fl", "are"],   # Cloudflare
    ["my", "company"],        # MyCompany — add your own
]
```

The `--prompt` flag is also effective for proper nouns — passing `--prompt "Cloudflare, WebAssembly"` biases the model toward recognizing them correctly before the merge step is needed.
