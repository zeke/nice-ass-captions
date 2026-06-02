#!/usr/bin/env python3
"""
caption.py — Burn stylish word-highlighted captions into videos.

Full pipeline: extract audio → transcribe with whisper.cpp → generate ASS → burn in with ffmpeg.

Usage:
    python caption.py input.mp4
    python caption.py input.mp4 --output out.mp4 --words 5 --model medium.en
"""

import re
import os
import sys
import shutil
import argparse
import tempfile
import subprocess
from pathlib import Path

# ---------------------------------------------------------------------------
# Style constants — tweak these to change the look
# ---------------------------------------------------------------------------

VIDEO_WIDTH   = 1920
VIDEO_HEIGHT  = 1080

FONT_NAME     = "Arial"
FONT_SIZE     = 72          # points; scale with video resolution

PAD_X         = 24          # horizontal padding inside box (pixels)
PAD_Y         = 28          # vertical padding inside box (pixels)
CORNER_R      = 18          # box corner radius (pixels)

MARGIN_BOTTOM = 80          # distance from bottom of frame to text baseline

# ASS alpha: 0x00 = fully opaque, 0xFF = fully transparent
ALPHA_DIM     = "&H99&"     # inactive words (~60% opaque)
ALPHA_BRIGHT  = "&H00&"     # active word (fully opaque)
COL_TEXT      = "&H00FFFFFF"  # caption text color (white, opaque)
COL_BOX       = "&H48000000"  # background box color (black, ~72% opaque)
MIN_CONTRAST  = 4.5         # WCAG AA contrast ratio for normal text

WORDS_PER_CHUNK = 5         # words per caption group

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
]

# ---------------------------------------------------------------------------
# Default model location
# ---------------------------------------------------------------------------

MODEL_DIR  = Path.home() / ".cache" / "nice-ass-captions"
MODEL_NAME = "ggml-medium.en.bin"

MODEL_SIZES = {
    "tiny.en":   "75MB",
    "base.en":   "142MB",
    "small.en":  "466MB",
    "medium.en": "1.5GB",
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

    # Auto-detect: look for any ggml-*.bin in MODEL_DIR, prefer medium.en
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
    print("Recommended: medium.en offers the best accuracy/speed tradeoff.")


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


def transcribe(wav_path, model_path, prompt, tmp_dir):
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

    subprocess.run(cmd, check=True, capture_output=True)
    return wts_stem + ".wts"


def _model_short_name(model_path):
    """Extract the model name for --dtw flag, e.g. 'medium.en' from 'ggml-medium.en.bin'."""
    name = Path(model_path).stem  # e.g. ggml-medium.en
    return name.replace("ggml-", "")


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
    CONTRACTION    = re.compile(r"^'[a-zA-Z]{1,2}$")  # 't, 's, 're, 've, 'll, 'd, 'm

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
    h  = cs // 360000;  cs %= 360000
    m  = cs // 6000;    cs %= 6000
    s  = cs // 100;     cs %= 100
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


def build_ass(chunks, video_width=VIDEO_WIDTH, video_height=VIDEO_HEIGHT, colors=None):
    colors = colors or {}
    header = ASS_HEADER.format(
        width   = video_width,
        height  = video_height,
        font    = FONT_NAME,
        size    = FONT_SIZE,
        text_col = colors.get("text", COL_TEXT),
        box_col = colors.get("box", COL_BOX),
    )

    cx     = video_width // 2
    text_y = video_height - MARGIN_BOTTOM
    pos_tag = rf"{{\an2\pos({cx},{text_y})}}"

    text_h = FONT_SIZE
    box_h  = text_h + PAD_Y * 2
    box_top = text_y - text_h - PAD_Y

    lines = []

    for chunk in chunks:
        chunk_words = chunk["words"]
        chunk_start = chunk["start"]
        chunk_end   = chunk["end"]
        chunk_start_ts = sec_to_ass(chunk_start)
        chunk_end_ts   = sec_to_ass(chunk_end)

        # Estimate box width from character counts
        space_w = FONT_SIZE * 0.28
        total_text_w = (
            sum(len(w["word"]) * FONT_SIZE * 0.52 for w in chunk_words)
            + space_w * (len(chunk_words) - 1)
        )
        box_w    = total_text_w + PAD_X * 2
        box_left = cx - box_w / 2

        # Layer 0: rounded background box
        drawing  = rounded_rect(box_left, box_top, box_w, box_h, CORNER_R)
        box_text = r"{\p1\an7\pos(0,0)}" + drawing + r"{\p0}"
        lines.append(f"Dialogue: 0,{chunk_start_ts},{chunk_end_ts},Box,,0,0,0,,{box_text}")

        # Layer 1: caption text — one line per chunk, \1a animated per word
        parts = []
        for i, w in enumerate(chunk_words):
            t_in  = max(0, int((w["start"] - chunk_start) * 1000))
            t_out = int((w["end"] - chunk_start) * 1000)

            if i == 0:
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

        caption_text = pos_tag + " ".join(parts)
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


def extract_palette(input_path, ffmpeg_path, samples=9, frame_size=64):
    """Sample frames and return (palette, pixels), where palette is frequent RGB buckets."""
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


def choose_palette_colors(input_path, ffmpeg_path, min_contrast=MIN_CONTRAST):
    """Choose global text and box colors from the video palette with accessible contrast."""
    palette, pixels = extract_palette(input_path, ffmpeg_path)
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


def rgb_to_hex(rgb):
    return "#" + "".join(f"{v:02X}" for v in rgb)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Burn stylish word-highlighted captions into a video."
    )
    parser.add_argument("input", help="Input video file")
    parser.add_argument("-o", "--output", help="Output video path (default: <input>-captioned.mp4)")
    parser.add_argument("--model", help="Whisper model path or name (e.g. medium.en)")
    parser.add_argument("--prompt", help="Initial prompt for whisper (improves proper noun accuracy)")
    parser.add_argument("--words", type=int, default=WORDS_PER_CHUNK, help=f"Words per caption chunk (default: {WORDS_PER_CHUNK})")
    parser.add_argument("--palette-colors", action="store_true", help="Derive text and background colors from the video palette")
    parser.add_argument("--keep-tmp", action="store_true", help="Keep intermediate audio/wts files")
    args = parser.parse_args()

    input_path = args.input
    if not os.path.isfile(input_path):
        print(f"Error: file not found: {input_path}")
        sys.exit(1)

    # Resolve output path
    if args.output:
        output_path = args.output
    else:
        stem = Path(input_path).stem
        output_path = str(Path(input_path).parent / f"{stem}-captioned.mp4")

    # Check dependencies
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

    model_path = resolve_model(args.model)

    # Detect video dimensions
    width, height = get_video_dimensions(input_path)
    print(f"Video: {width}x{height}")
    print(f"Model: {model_path}")
    print(f"Output: {output_path}")
    colors = None
    if args.palette_colors:
        print("Sampling video palette...")
        colors = choose_palette_colors(input_path, ffmpeg_full)
        fallback = " fallback" if colors.get("fallback") else ""
        print(
            "Palette colors: "
            f"text {rgb_to_hex(colors['text_rgb'])}, "
            f"box {rgb_to_hex(colors['box_rgb'])} "
            f"({100 * (1 - colors['box_alpha'] / 255):.0f}% opaque), "
            f"contrast {colors['contrast']:.1f}:1{fallback}"
        )
    print()

    tmp_dir = tempfile.mkdtemp(prefix="nice-ass-captions-")
    try:
        print("Extracting audio...")
        wav_path = extract_audio(input_path, tmp_dir)

        print("Transcribing with whisper.cpp...")
        wts_path = transcribe(wav_path, model_path, args.prompt, tmp_dir)

        print("Parsing word timestamps...")
        words = parse_wts(wts_path)
        print(f"  {len(words)} words found")

        chunks = chunk_words(words, args.words)
        print(f"  {len(chunks)} chunks of ~{args.words} words")

        print("Generating ASS captions...")
        ass_path = os.path.join(tmp_dir, "captions.ass")
        ass = build_ass(chunks, video_width=width, video_height=height, colors=colors)
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
