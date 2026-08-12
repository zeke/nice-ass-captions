#!/usr/bin/env python3
"""
caption.py — Burn stylish word-highlighted captions into videos.

Full pipeline: extract audio → transcribe with whisper.cpp → generate ASS → burn in with ffmpeg.

Usage:
    python caption.py input.mp4
    python caption.py input.mp4 --output out.mp4 --words 5 --model large-v3-turbo
"""

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

# ---------------------------------------------------------------------------
# Style constants — tweak these to change the look
# ---------------------------------------------------------------------------

VIDEO_WIDTH   = 1920
VIDEO_HEIGHT  = 1080

FONT_NAME     = "Arial"
FONT_SIZE     = 44          # points; scale with video resolution

PAD_X         = 15          # horizontal padding inside box (pixels)
PAD_Y         = 17          # vertical padding inside box (pixels)
CORNER_R      = 11          # box corner radius (pixels)

MARGIN_BOTTOM = 80          # distance from bottom of frame to text baseline

MAX_CAPTION_WIDTH_RATIO = 0.90  # max caption box width, as a fraction of video width
LINE_SPACING            = 1.25  # vertical spacing between wrapped lines, as a multiple of FONT_SIZE

# ASS alpha: 0x00 = fully opaque, 0xFF = fully transparent
ALPHA_DIM     = "&H99&"     # inactive words (~60% opaque)
ALPHA_BRIGHT  = "&H00&"     # active word (fully opaque)
COL_TEXT      = "&H00FFFFFF"  # caption text color (white, opaque)
COL_BOX       = "&H48000000"  # background box color (black, ~72% opaque)
MIN_CONTRAST  = 4.5         # WCAG AA contrast ratio for normal text

WORDS_PER_CHUNK = 5         # words per caption group
CAPTION_POSITION = "bottom"  # top, center, or bottom

# ---------------------------------------------------------------------------
# Compound word merges
# whisper.cpp tokenizes some words at subword boundaries. Add entries here
# as you encounter new ones. Each entry is a list of lowercase tokens that
# should be joined (without spaces) into a single word.
# ---------------------------------------------------------------------------

COMPOUND_MERGES = [
    ["cloud", "fl", "are"],    # Cloudflare
    ["cloud", "flare"],        # Cloudflare (alternate)
    ["b", "rows", "ers"],      # Browsers
    ["b", "rowser"],           # Browser
    ["b", "rowse"],            # browse
    ["b", "rows", "ing"],      # browsing
    ["b", "rows", "ed"],       # browsed
    ["cla", "ude"],            # Claude (tokenizer split A)
    ["claud", "e"],            # Claude (tokenizer split B)
    ["chat", "g", "p", "t"],   # ChatGPT
    ["code", "x"],             # Codex
    ["blo", "at"],             # bloat
    ["blo", "ated"],           # bloated
    ["un", "ad", "ul", "ter", "ated"],  # unadulterated
]

# ---------------------------------------------------------------------------
# Default model location
# ---------------------------------------------------------------------------

MODEL_DIR  = Path.home() / ".cache" / "nice-ass-captions"
MODEL_NAME = "ggml-large-v3-turbo.bin"

MODEL_SIZES = {
    "tiny.en":        "75MB",
    "base.en":        "142MB",
    "small.en":       "466MB",
    "medium.en":      "1.5GB",
    "large-v3-turbo": "1.6GB",
    "large-v3":       "3.1GB",
}

# ---------------------------------------------------------------------------
# Dependency detection
# ---------------------------------------------------------------------------

def find_whisper_cli():
    """Return path to whisper-cli or None."""
    return shutil.which("whisper-cli")


def find_ffmpeg_with_libass():
    """
    Return path to an ffmpeg binary that has libass (subtitle burn-in support).
    Checks ffmpeg-full Homebrew paths first, then PATH.
    """
    candidates = [
        "/opt/homebrew/opt/ffmpeg-full/bin/ffmpeg",  # Apple Silicon
        "/usr/local/opt/ffmpeg-full/bin/ffmpeg",      # Intel Mac
        shutil.which("ffmpeg"),
    ]
    for path in candidates:
        if not path or not os.path.isfile(path):
            continue
        result = subprocess.run(
            [path, "-filters"],
            capture_output=True, text=True
        )
        if "subtitles" in result.stdout or "subtitles" in result.stderr:
            return path
    return None


def find_ffmpeg_basic():
    """Return any ffmpeg for audio extraction (doesn't need libass)."""
    return shutil.which("ffmpeg")


def resolve_model(model_arg):
    """
    Resolve the model path from a flag value or auto-detect.
    Accepts: full path, short name like 'medium.en', or None (auto-detect).
    """
    if model_arg:
        p = Path(model_arg)
        if p.exists():
            return str(p)
        # Treat as a short name
        p = MODEL_DIR / f"ggml-{model_arg}.bin"
        if p.exists():
            return str(p)
        print(f"Model not found: {model_arg}")
        print_model_download_help()
        sys.exit(1)

    # Auto-detect: look for any ggml-*.bin in MODEL_DIR, prefer large-v3-turbo
    if MODEL_DIR.exists():
        preferred = MODEL_DIR / MODEL_NAME
        if preferred.exists():
            return str(preferred)
        candidates = sorted(MODEL_DIR.glob("ggml-*.bin"))
        if candidates:
            return str(candidates[-1])

    print(f"No whisper model found in {MODEL_DIR}")
    print_model_download_help()
    sys.exit(1)


def print_model_download_help():
    print()
    print("Download a model with:")
    print(f"  mkdir -p {MODEL_DIR}")
    for name, size in MODEL_SIZES.items():
        print(f"  # {name} ({size})")
        print(f"  curl -L -o {MODEL_DIR}/ggml-{name}.bin \\")
        print(f"    https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-{name}.bin")
    print()
    print("Recommended: large-v3-turbo offers the best accuracy/speed tradeoff.")


# ---------------------------------------------------------------------------
# Pipeline steps
# ---------------------------------------------------------------------------

def extract_audio(input_path, tmp_dir):
    """Extract 16kHz mono WAV from input video."""
    wav_path = os.path.join(tmp_dir, "audio.wav")
    ffmpeg = find_ffmpeg_basic()
    subprocess.run([
        ffmpeg, "-y", "-i", input_path,
        "-ar", "16000", "-ac", "1", "-c:a", "pcm_s16le",
        wav_path
    ], check=True, capture_output=True)
    return wav_path


def transcribe(wav_path, model_path, prompt, tmp_dir, carry_initial_prompt=False):
    """
    Run whisper-cli with DTW word-level timestamps.
    Returns path to the .wts bash script containing per-word timing.
    """
    whisper = find_whisper_cli()
    wts_stem = os.path.join(tmp_dir, "transcript")

    cmd = [
        whisper,
        "-m", model_path,
        "-f", wav_path,
        "--language", "en",
        "--dtw", _model_short_name(model_path),
        "--output-words",
        "--output-file", wts_stem,
    ]
    if prompt:
        cmd += ["--prompt", prompt]
        if carry_initial_prompt:
            cmd += ["--carry-initial-prompt"]

    subprocess.run(cmd, check=True, capture_output=True)
    return wts_stem + ".wts"


def read_transcript_prompt(transcript_path):
    """Read raw transcript text for use as a whisper initial prompt."""
    try:
        return Path(transcript_path).read_text(encoding="utf-8").strip()
    except FileNotFoundError:
        raise ValueError(f"transcript file not found: {transcript_path}") from None
    except IsADirectoryError:
        raise ValueError(f"transcript path is a directory: {transcript_path}") from None
    except UnicodeDecodeError as e:
        raise ValueError(f"transcript file is not valid UTF-8: {transcript_path}") from e


def combine_prompts(prompt, transcript_prompt):
    parts = [part.strip() for part in (prompt, transcript_prompt) if part and part.strip()]
    return "\n\n".join(parts)


def _model_short_name(model_path):
    """Extract the model name for --dtw flag, e.g. 'medium.en' from 'ggml-medium.en.bin'.

    whisper.cpp's --dtw presets for large models are dotted (large.v3, large.v3.turbo)
    while the model filenames are hyphenated (ggml-large-v3-turbo.bin), so convert those.
    """
    name = Path(model_path).stem.replace("ggml-", "")  # e.g. medium.en, large-v3-turbo
    if name.startswith("large-"):
        name = name.replace("-", ".")  # large-v3-turbo -> large.v3.turbo
    return name


# ---------------------------------------------------------------------------
# Parse .wts for word-level timestamps
# ---------------------------------------------------------------------------

def parse_wts(wts_path):
    """
    Extract word-timed tokens from a whisper.cpp .wts bash script.
    The .wts file contains ffmpeg drawtext filters; lightgreen entries
    mark each active word with its timing window.
    Returns a list of {word, start, end} dicts.
    """
    raw = open(wts_path).read()
    unescaped = raw.replace("\\'", "'").replace("\\\\", "\\")

    pattern = (
        r"fontcolor=lightgreen:x=[^:]+:y=[^:]+:"
        r"text='([^']*)':enable='between\(t,([\d.]+),([\d.]+)\)'"
    )
    matches = re.findall(pattern, unescaped)

    raw_tokens = []
    seen = set()

    for raw_text, start, end in matches:
        stripped = raw_text.strip()
        if re.match(r'^[\s\\_]+$', stripped):
            continue
        if stripped.startswith("\\"):
            continue

        before_pipe = raw_text.split("|")[0]
        tokens = [t for t in before_pipe.split() if t not in (">", "\\", "")]
        if not tokens:
            continue

        word = tokens[-1].lstrip(">\"")
        if not word:
            continue

        key = (word, start)
        if key in seen:
            continue
        seen.add(key)

        raw_tokens.append({"word": word, "start": float(start), "end": float(end)})

    # Merge compound words split by whisper's tokenizer (e.g. Cloud+fl+are → Cloudflare)
    for compound in COMPOUND_MERGES:
        i = 0
        merged = []
        while i < len(raw_tokens):
            seq_len = len(compound)
            if i + seq_len <= len(raw_tokens):
                window = [t["word"].lower().rstrip(".,!?;:") for t in raw_tokens[i:i+seq_len]]
                if window == compound:
                    joined = "".join(t["word"] for t in raw_tokens[i:i+seq_len])
                    merged.append({
                        "word": joined,
                        "start": raw_tokens[i]["start"],
                        "end":   raw_tokens[i+seq_len-1]["end"],
                    })
                    i += seq_len
                    continue
            merged.append(raw_tokens[i])
            i += 1
        raw_tokens = merged

    # Merge trailing punctuation and contraction suffixes into preceding word
    TRAILING_PUNCT = re.compile(r'^[,\.!\?;:\-]+$')
    CONTRACTION    = re.compile(r"^['\u2018\u2019][a-zA-Z]{1,2}$")  # 't, 's, 're, 've, 'll, 'd, 'm — straight or curly apostrophe

    words = []
    for tok in raw_tokens:
        w = tok["word"]
        if (TRAILING_PUNCT.match(w) or CONTRACTION.match(w)) and words:
            words[-1]["word"] += w
            words[-1]["end"] = tok["end"]
        elif re.search(r"[A-Za-z0-9]", w):
            words.append(tok)

    return words


# ---------------------------------------------------------------------------
# Chunk words
# ---------------------------------------------------------------------------

def chunk_words(words, chunk_size):
    chunks = []
    for i in range(0, len(words), chunk_size):
        group = words[i:i+chunk_size]
        chunks.append({
            "words": group,
            "start": group[0]["start"],
            "end":   group[-1]["end"],
        })
    return chunks


# ---------------------------------------------------------------------------
# Time helpers
# ---------------------------------------------------------------------------

def sec_to_ass(seconds):
    """Float seconds -> ASS timestamp H:MM:SS.cc"""
    cs = int(round(seconds * 100))
    h, cs = divmod(cs, 360000)
    m, cs = divmod(cs, 6000)
    s, cs = divmod(cs, 100)
    return f"{h}:{m:02d}:{s:02d}.{cs:02d}"


# ---------------------------------------------------------------------------
# Rounded rectangle drawing command
# ---------------------------------------------------------------------------

def rounded_rect(x, y, w, h, r):
    """ASS vector drawing commands for a rounded rectangle."""
    k = r * 0.5523  # Bezier control point offset for quarter-circle approximation
    return (
        f"m {int(x+r)} {int(y)} "
        f"l {int(x+w-r)} {int(y)} "
        f"b {int(x+w-r+k)} {int(y)} {int(x+w)} {int(y+r-k)} {int(x+w)} {int(y+r)} "
        f"l {int(x+w)} {int(y+h-r)} "
        f"b {int(x+w)} {int(y+h-r+k)} {int(x+w-r+k)} {int(y+h)} {int(x+w-r)} {int(y+h)} "
        f"l {int(x+r)} {int(y+h)} "
        f"b {int(x+r-k)} {int(y+h)} {int(x)} {int(y+h-r+k)} {int(x)} {int(y+h-r)} "
        f"l {int(x)} {int(y+r)} "
        f"b {int(x)} {int(y+r-k)} {int(x+r-k)} {int(y)} {int(x+r)} {int(y)}"
    )


# ---------------------------------------------------------------------------
# ASS generation
# ---------------------------------------------------------------------------

ASS_HEADER = """\
[Script Info]
ScriptType: v4.00+
PlayResX: {width}
PlayResY: {height}
WrapStyle: 0
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Caption,{font},{size},{text_col},{text_col},&H00000000,&H00000000,-1,0,0,0,100,100,0,0,1,0,0,2,0,0,0,1
Style: Box,Arial,1,{box_col},{box_col},{box_col},{box_col},0,0,0,0,100,100,0,0,1,0,0,7,0,0,0,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""


def get_caption_layout(position, video_width, video_height):
    cx = video_width // 2

    if position == "top":
        text_y = MARGIN_BOTTOM + PAD_Y
        return "8", cx, text_y
    if position == "center":
        text_y = video_height // 2
        return "5", cx, text_y
    if position == "bottom":
        text_y = video_height - MARGIN_BOTTOM
        return "2", cx, text_y
    raise ValueError(f"Invalid caption position: {position}")


def compute_box_top(position, text_y, box_h, video_height):
    """Box top position, accounting for a variable box height (multi-line
    chunks). Each anchor grows the box in the direction away from its edge:
    top-anchored boxes grow downward, bottom-anchored boxes grow upward,
    center-anchored boxes grow symmetrically."""
    if position == "top":
        return text_y - PAD_Y
    if position == "center":
        return (video_height - box_h) / 2
    if position == "bottom":
        return text_y - box_h + PAD_Y
    raise ValueError(f"Invalid caption position: {position}")


def estimate_word_width(word, font_size=FONT_SIZE):
    return len(word) * font_size * 0.52


def wrap_chunk_words(chunk_words, max_text_width, font_size=FONT_SIZE):
    """Greedily pack words into lines that fit within max_text_width."""
    space_w = font_size * 0.28
    lines = []
    current = []
    current_w = 0.0
    for w in chunk_words:
        word_w = estimate_word_width(w["word"], font_size)
        added_w = word_w if not current else word_w + space_w
        if current and current_w + added_w > max_text_width:
            lines.append(current)
            current = [w]
            current_w = word_w
        else:
            current.append(w)
            current_w += added_w
    if current:
        lines.append(current)
    return lines


def line_width(line, font_size=FONT_SIZE):
    space_w = font_size * 0.28
    return sum(estimate_word_width(w["word"], font_size) for w in line) + space_w * (len(line) - 1)


def build_ass(chunks, video_width=VIDEO_WIDTH, video_height=VIDEO_HEIGHT, colors=None, position=CAPTION_POSITION, box=True, font_size=FONT_SIZE):
    colors = colors or {}
    header = ASS_HEADER.format(
        width   = video_width,
        height  = video_height,
        font    = FONT_NAME,
        size    = font_size,
        text_col = colors.get("text", COL_TEXT),
        box_col = colors.get("box", COL_BOX),
    )

    align, cx, text_y = get_caption_layout(position, video_width, video_height)
    pos_tag = rf"{{\an{align}\pos({cx},{text_y})}}"

    max_text_width = video_width * MAX_CAPTION_WIDTH_RATIO - PAD_X * 2

    lines = []

    for chunk in chunks:
        chunk_words = chunk["words"]
        chunk_start = chunk["start"]
        chunk_end   = chunk["end"]
        chunk_start_ts = sec_to_ass(chunk_start)
        chunk_end_ts   = sec_to_ass(chunk_end)

        # Wrap words onto multiple lines if the chunk is too wide for the frame
        word_lines = wrap_chunk_words(chunk_words, max_text_width, font_size=font_size)

        box_w    = max(line_width(wl, font_size=font_size) for wl in word_lines) + PAD_X * 2
        box_left = cx - box_w / 2

        line_height = font_size * LINE_SPACING
        text_h = line_height * len(word_lines)
        box_h  = text_h + PAD_Y * 2
        box_top = compute_box_top(position, text_y, box_h, video_height)

        # Per-chunk color overrides (--colorize per-chunk). Empty for global/off,
        # where colors come from the style header instead.
        chunk_colors = chunk.get("colors")
        if chunk_colors:
            box_override = (
                rf"\1c{rgb_to_ass_color(chunk_colors['box_rgb'])}"
                rf"\1a&H{chunk_colors['box_alpha']:02X}&"
            )
            text_override = rf"\1c{rgb_to_ass_color(chunk_colors['text_rgb'])}"
        else:
            box_override = ""
            text_override = ""

        # Layer 0: rounded background box
        if box:
            drawing  = rounded_rect(box_left, box_top, box_w, box_h, CORNER_R)
            box_text = r"{" + box_override + r"\p1\an7\pos(0,0)}" + drawing + r"{\p0}"
            lines.append(f"Dialogue: 0,{chunk_start_ts},{chunk_end_ts},Box,,0,0,0,,{box_text}")

        # Layer 1: caption text — one or more lines per chunk, \1a animated per word
        rendered_lines = []
        for word_line in word_lines:
            parts = []
            for w in word_line:
                is_first_word_overall = w is chunk_words[0]
                t_in  = max(0, int((w["start"] - chunk_start) * 1000))
                t_out = int((w["end"] - chunk_start) * 1000)

                if is_first_word_overall:
                    # First word starts bright (t=0 snap is unreliable in libass)
                    word_tags = (
                        rf"{{\1a{ALPHA_BRIGHT}"
                        rf"\t({t_out},{t_out},\1a{ALPHA_DIM})}}"
                    )
                else:
                    word_tags = (
                        rf"{{\1a{ALPHA_DIM}"
                        rf"\t({t_in},{t_in},\1a{ALPHA_BRIGHT})"
                        rf"\t({t_out},{t_out},\1a{ALPHA_DIM})}}"
                    )
                parts.append(word_tags + w["word"])
            rendered_lines.append(" ".join(parts))

        caption_prefix = (rf"{{{text_override}}}" if text_override else "") + pos_tag
        caption_text = caption_prefix + r"\N".join(rendered_lines)
        lines.append(f"Dialogue: 1,{chunk_start_ts},{chunk_end_ts},Caption,,0,0,0,,{caption_text}")

    return header + "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Burn captions into video
# ---------------------------------------------------------------------------

def burn_captions(input_path, ass_path, output_path, ffmpeg_path):
    subprocess.run([
        ffmpeg_path, "-y",
        "-i", input_path,
        "-vf", f"ass={ass_path}",
        "-c:a", "copy",
        output_path,
    ], check=True, capture_output=True)


# ---------------------------------------------------------------------------
# Video dimension detection
# ---------------------------------------------------------------------------

def get_video_dimensions(input_path):
    """Return (width, height) of input video using ffprobe."""
    result = subprocess.run([
        "ffprobe", "-v", "quiet",
        "-select_streams", "v:0",
        "-show_entries", "stream=width,height",
        "-of", "csv=p=0",
        input_path,
    ], capture_output=True, text=True)
    parts = result.stdout.strip().split(",")
    if len(parts) == 2:
        return int(parts[0]), int(parts[1])
    return VIDEO_WIDTH, VIDEO_HEIGHT


def get_video_duration(input_path):
    """Return duration in seconds using ffprobe, or 0 if unavailable."""
    result = subprocess.run([
        "ffprobe", "-v", "quiet",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        input_path,
    ], capture_output=True, text=True)
    try:
        return float(result.stdout.strip())
    except ValueError:
        return 0


# ---------------------------------------------------------------------------
# Palette-derived colors
# ---------------------------------------------------------------------------

def rgb_to_ass(rgb, alpha=0):
    """RGB tuple + ASS alpha byte -> ASS &HAABBGGRR color."""
    r, g, b = [max(0, min(255, int(v))) for v in rgb]
    return f"&H{alpha:02X}{b:02X}{g:02X}{r:02X}"


def rgb_to_ass_color(rgb):
    r"""RGB tuple -> ASS \1c color literal &HBBGGRR&."""
    r, g, b = [max(0, min(255, int(v))) for v in rgb]
    return f"&H{b:02X}{g:02X}{r:02X}&"


def srgb_to_linear(c):
    c = c / 255
    if c <= 0.03928:
        return c / 12.92
    return ((c + 0.055) / 1.055) ** 2.4


def relative_luminance(rgb):
    r, g, b = [srgb_to_linear(v) for v in rgb]
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def contrast_ratio(a, b):
    l1 = relative_luminance(a)
    l2 = relative_luminance(b)
    lighter = max(l1, l2)
    darker = min(l1, l2)
    return (lighter + 0.05) / (darker + 0.05)


def saturation(rgb):
    return (max(rgb) - min(rgb)) / 255


def mix(a, b, amount):
    return tuple(round(a[i] * (1 - amount) + b[i] * amount) for i in range(3))


def composite_box_over_video(box_rgb, box_alpha, video_rgb):
    opacity = 1 - (box_alpha / 255)
    return tuple(round(box_rgb[i] * opacity + video_rgb[i] * (1 - opacity)) for i in range(3))


def extract_palette(input_path, ffmpeg_path, samples=9, frame_size=64, timestamps=None):
    """Sample frames and return (palette, pixels), where palette is frequent RGB buckets.

    If timestamps is given, those exact frame times (seconds) are sampled. Otherwise
    samples are spread evenly across the whole video.
    """
    if timestamps is None:
        duration = get_video_duration(input_path)
        if duration > 0:
            timestamps = [duration * (i + 1) / (samples + 1) for i in range(samples)]
        else:
            timestamps = [0]

    counts = {}
    pixels = []
    expected = frame_size * frame_size * 3

    for timestamp in timestamps:
        result = subprocess.run([
            ffmpeg_path, "-v", "error",
            "-ss", f"{timestamp:.3f}",
            "-i", input_path,
            "-frames:v", "1",
            "-vf", f"scale={frame_size}:{frame_size}:flags=bilinear,format=rgb24",
            "-f", "rawvideo", "-",
        ], capture_output=True)
        raw = result.stdout
        if len(raw) != expected:
            continue

        for i in range(0, len(raw), 3):
            rgb = (raw[i], raw[i + 1], raw[i + 2])
            pixels.append(rgb)
            bucket = tuple((v // 16) * 16 + 8 for v in rgb)
            counts[bucket] = counts.get(bucket, 0) + 1

    ranked = sorted(
        counts.items(),
        key=lambda item: item[1] * (1 + saturation(item[0])),
        reverse=True,
    )
    return [rgb for rgb, _ in ranked[:32]], pixels


def min_contrast_over_samples(text_rgb, box_rgb, box_alpha, pixels):
    if not pixels:
        return contrast_ratio(text_rgb, box_rgb)
    return min(
        contrast_ratio(text_rgb, composite_box_over_video(box_rgb, box_alpha, pixel))
        for pixel in pixels[::max(1, len(pixels) // 512)]
    )


def palette_pair_score(text_rgb, box_rgb, box_alpha, contrast):
    """Prefer visibly colored palette pairs after contrast has passed."""
    contrast_bonus = min(contrast - MIN_CONTRAST, 4) * 0.15
    alpha_penalty = abs(box_alpha - 0x48) / 255
    near_white_penalty = 0.55 if min(text_rgb) > 232 and saturation(text_rgb) < 0.16 else 0
    near_black_penalty = 0.35 if relative_luminance(box_rgb) < 0.012 and saturation(box_rgb) < 0.16 else 0
    return (
        saturation(text_rgb) * 3
        + saturation(box_rgb) * 1.2
        + relative_luminance(text_rgb) * 0.45
        + contrast_bonus
        - alpha_penalty
        - near_white_penalty
        - near_black_penalty
    )


def select_colors(palette, pixels, min_contrast=MIN_CONTRAST):
    """Choose a text/box color pair from a palette with accessible contrast."""
    if not palette:
        return {
            "text": COL_TEXT,
            "box": COL_BOX,
            "text_rgb": (255, 255, 255),
            "box_rgb": (0, 0, 0),
            "box_alpha": 0x48,
            "contrast": min_contrast,
            "fallback": True,
        }

    palette = palette[:16]

    bg_candidates = []
    for rgb in palette:
        bg_candidates.append(rgb)
        bg_candidates.append(mix(rgb, (0, 0, 0), 0.45))
        bg_candidates.append(mix(rgb, (0, 0, 0), 0.70))
    bg_candidates = sorted(
        dict.fromkeys(bg_candidates),
        key=lambda rgb: (relative_luminance(rgb), -saturation(rgb)),
    )

    text_candidates = []
    for rgb in palette:
        text_candidates.append(rgb)
        text_candidates.append(mix(rgb, (255, 255, 255), 0.55))
        text_candidates.append(mix(rgb, (255, 255, 255), 0.78))
    text_candidates.extend([(255, 255, 255), (245, 245, 235), (0, 0, 0)])
    text_candidates = sorted(
        dict.fromkeys(text_candidates),
        key=lambda rgb: (abs(relative_luminance(rgb) - 0.85), -saturation(rgb)),
    )

    best = None
    best_passing = None
    for box_alpha in (0x48, 0x40, 0x38, 0x30, 0x28):
        for box_rgb in bg_candidates:
            for text_rgb in text_candidates:
                contrast = min_contrast_over_samples(text_rgb, box_rgb, box_alpha, pixels)
                score = palette_pair_score(text_rgb, box_rgb, box_alpha, contrast)
                if best is None or contrast > best["contrast"]:
                    best = {
                        "text_rgb": text_rgb,
                        "box_rgb": box_rgb,
                        "box_alpha": box_alpha,
                        "contrast": contrast,
                    }
                if contrast >= min_contrast:
                    if best_passing is None or score > best_passing["score"]:
                        best_passing = {
                            "text_rgb": text_rgb,
                            "box_rgb": box_rgb,
                            "box_alpha": box_alpha,
                            "contrast": contrast,
                            "score": score,
                        }

    if best_passing:
        text_rgb = best_passing["text_rgb"]
        box_rgb = best_passing["box_rgb"]
        box_alpha = best_passing["box_alpha"]
        return {
            "text": rgb_to_ass(text_rgb),
            "box": rgb_to_ass(box_rgb, box_alpha),
            "text_rgb": text_rgb,
            "box_rgb": box_rgb,
            "box_alpha": box_alpha,
            "contrast": best_passing["contrast"],
            "fallback": False,
        }

    if best:
        text_rgb = best["text_rgb"]
        box_rgb = best["box_rgb"]
        box_alpha = best["box_alpha"]
        contrast = best["contrast"]
        if contrast >= min_contrast:
            return {
                "text": rgb_to_ass(text_rgb),
                "box": rgb_to_ass(box_rgb, box_alpha),
                "text_rgb": text_rgb,
                "box_rgb": box_rgb,
                "box_alpha": box_alpha,
                "contrast": contrast,
                "fallback": False,
            }

    # Keep some transparency, but force a readable pair if palette colors fail.
    text_rgb = (255, 255, 255)
    box_rgb = mix(bg_candidates[0], (0, 0, 0), 0.85)
    box_alpha = 0x28
    contrast = min_contrast_over_samples(text_rgb, box_rgb, box_alpha, pixels)
    if contrast < min_contrast:
        box_rgb = (0, 0, 0)
        contrast = min_contrast_over_samples(text_rgb, box_rgb, box_alpha, pixels)

    return {
        "text": rgb_to_ass(text_rgb),
        "box": rgb_to_ass(box_rgb, box_alpha),
        "text_rgb": text_rgb,
        "box_rgb": box_rgb,
        "box_alpha": box_alpha,
        "contrast": contrast,
        "fallback": True,
    }


def choose_palette_colors(input_path, ffmpeg_path, min_contrast=MIN_CONTRAST):
    """Choose one global text/box pair from the whole-video palette."""
    palette, pixels = extract_palette(input_path, ffmpeg_path)
    return select_colors(palette, pixels, min_contrast)


def choose_chunk_colors(chunks, input_path, ffmpeg_path, min_contrast=MIN_CONTRAST):
    """Attach a text/box pair to each chunk, sampled from the frame at its midpoint."""
    for chunk in chunks:
        midpoint = (chunk["start"] + chunk["end"]) / 2
        palette, pixels = extract_palette(
            input_path, ffmpeg_path, timestamps=[midpoint]
        )
        chunk["colors"] = select_colors(palette, pixels, min_contrast)
    return chunks


def rgb_to_hex(rgb):
    return "#" + "".join(f"{v:02X}" for v in rgb)


def format_colors(colors):
    """Human-readable summary of a chosen color pair for console output."""
    fallback = " fallback" if colors.get("fallback") else ""
    return (
        f"text {rgb_to_hex(colors['text_rgb'])}, "
        f"box {rgb_to_hex(colors['box_rgb'])} "
        f"({100 * (1 - colors['box_alpha'] / 255):.0f}% opaque), "
        f"contrast {colors['contrast']:.1f}:1{fallback}"
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Burn stylish word-highlighted captions into a video."
    )
    parser.add_argument("input", help="Input video file")
    parser.add_argument("-o", "--output", help="Output video path (default: <input>-captioned.mp4)")
    parser.add_argument("--model", help="Whisper model path or name (e.g. large-v3-turbo)")
    parser.add_argument("--prompt", help="Initial prompt for whisper (improves proper noun accuracy)")
    parser.add_argument("--transcript", help="Raw transcript text file to use as a whisper prompt")
    parser.add_argument("--words-json", help="Path to a JSON file of pre-computed [{word, start, end}, ...] word timings. Skips whisper.cpp transcription entirely.")
    parser.add_argument("--words", type=int, default=WORDS_PER_CHUNK, help=f"Words per caption chunk (default: {WORDS_PER_CHUNK})")
    parser.add_argument("--position", choices=("top", "center", "bottom"), default=CAPTION_POSITION, help=f"Caption position (default: {CAPTION_POSITION})")
    parser.add_argument("--colorize", choices=("global", "per-chunk"), default=None, help="Derive colors from the video imagery: 'global' (one pair) or 'per-chunk' (per caption block)")
    parser.add_argument("--keep-tmp", action="store_true", help="Keep intermediate audio/wts files")
    parser.add_argument("--no-box", action="store_true", help="Don't draw the background box behind captions")
    parser.add_argument("--font-size", type=int, default=FONT_SIZE, help=f"Caption font size in points (default: {FONT_SIZE})")
    args = parser.parse_args()

    input_path = args.input
    if not os.path.isfile(input_path):
        print(f"Error: file not found: {input_path}")
        sys.exit(1)

    transcript_prompt = None
    if args.transcript:
        try:
            transcript_prompt = read_transcript_prompt(args.transcript)
        except ValueError as e:
            print(f"Error: {e}")
            sys.exit(1)

    prompt = combine_prompts(args.prompt, transcript_prompt)

    # Resolve output path
    if args.output:
        output_path = args.output
    else:
        stem = Path(input_path).stem
        output_path = str(Path(input_path).parent / f"{stem}-captioned.mp4")

    # Check dependencies
    if not args.words_json:
        whisper = find_whisper_cli()
        if not whisper:
            print("Error: whisper-cli not found in PATH.")
            print("Install with: brew install whisper-cpp")
            sys.exit(1)

    ffmpeg_full = find_ffmpeg_with_libass()
    if not ffmpeg_full:
        print("Error: no ffmpeg with libass support found.")
        print("Install with: brew install ffmpeg-full")
        sys.exit(1)

    model_path = None if args.words_json else resolve_model(args.model)

    # Detect video dimensions
    width, height = get_video_dimensions(input_path)
    print(f"Video: {width}x{height}")
    if args.words_json:
        print(f"Words: {args.words_json} (skipping whisper transcription)")
    else:
        print(f"Model: {model_path}")
    print(f"Output: {output_path}")
    if args.transcript:
        print(f"Transcript prompt: {args.transcript}")
    colors = None
    if args.colorize == "global":
        print("Sampling video palette...")
        colors = choose_palette_colors(input_path, ffmpeg_full)
        print("Colorize: " + format_colors(colors))
    print()

    tmp_dir = tempfile.mkdtemp(prefix="nice-ass-captions-")
    try:
        if args.words_json:
            print("Loading pre-computed word timings...")
            with open(args.words_json, encoding="utf-8") as f:
                words = json.load(f)
        else:
            print("Extracting audio...")
            wav_path = extract_audio(input_path, tmp_dir)

            print("Transcribing with whisper.cpp...")
            wts_path = transcribe(wav_path, model_path, prompt, tmp_dir, carry_initial_prompt=bool(transcript_prompt))

            print("Parsing word timestamps...")
            words = parse_wts(wts_path)
        print(f"  {len(words)} words found")

        chunks = chunk_words(words, args.words)
        print(f"  {len(chunks)} chunks of ~{args.words} words")

        if args.colorize == "per-chunk":
            print("Sampling per-chunk colors...")
            choose_chunk_colors(chunks, input_path, ffmpeg_full)
            for n, chunk in enumerate(chunks, 1):
                print(f"  block {n}: " + format_colors(chunk["colors"]))

        print("Generating ASS captions...")
        ass_path = os.path.join(tmp_dir, "captions.ass")
        ass = build_ass(chunks, video_width=width, video_height=height, colors=colors, position=args.position, box=not args.no_box, font_size=args.font_size)
        with open(ass_path, "w", encoding="utf-8") as f:
            f.write(ass)

        print("Burning captions into video...")
        burn_captions(input_path, ass_path, output_path, ffmpeg_full)

        print()
        print(f"Done: {output_path}")

        if args.keep_tmp:
            print(f"Temp files: {tmp_dir}")
        else:
            shutil.rmtree(tmp_dir, ignore_errors=True)

    except subprocess.CalledProcessError as e:
        print(f"\nError running command: {e.cmd}")
        if e.stderr:
            print(e.stderr.decode(errors="replace")[-2000:])
        shutil.rmtree(tmp_dir, ignore_errors=True)
        sys.exit(1)
    except Exception as e:
        print(f"\nError: {e}")
        shutil.rmtree(tmp_dir, ignore_errors=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
