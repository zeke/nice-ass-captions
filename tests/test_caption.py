"""Lightweight unit tests for caption.py — pure functions only, no ffmpeg/whisper."""

import pytest

import caption

# ---------------------------------------------------------------------------
# Color conversion helpers
# ---------------------------------------------------------------------------

def test_rgb_to_ass_default_alpha():
    # &HAABBGGRR — opaque (alpha 00), R=10 G=20 B=30
    assert caption.rgb_to_ass((0x10, 0x20, 0x30)) == "&H00302010"


def test_rgb_to_ass_with_alpha_and_clamping():
    assert caption.rgb_to_ass((300, -5, 128), alpha=0x48) == "&H488000FF"


def test_rgb_to_ass_color_literal():
    # \1c literal is &HBBGGRR&
    assert caption.rgb_to_ass_color((245, 233, 200)) == "&HC8E9F5&"


def test_rgb_to_hex():
    assert caption.rgb_to_hex((26, 20, 11)) == "#1A140B"


# ---------------------------------------------------------------------------
# Contrast / luminance math
# ---------------------------------------------------------------------------

def test_contrast_ratio_extremes():
    assert caption.contrast_ratio((255, 255, 255), (0, 0, 0)) == 21.0
    assert caption.contrast_ratio((100, 100, 100), (100, 100, 100)) == 1.0


def test_relative_luminance_bounds():
    assert caption.relative_luminance((0, 0, 0)) == 0.0
    assert abs(caption.relative_luminance((255, 255, 255)) - 1.0) < 1e-9


def test_saturation():
    assert caption.saturation((128, 128, 128)) == 0.0
    assert caption.saturation((255, 0, 0)) == 1.0


def test_mix_endpoints_and_midpoint():
    assert caption.mix((0, 0, 0), (255, 255, 255), 0) == (0, 0, 0)
    assert caption.mix((0, 0, 0), (255, 255, 255), 1) == (255, 255, 255)
    assert caption.mix((0, 0, 0), (200, 100, 50), 0.5) == (100, 50, 25)


def test_composite_box_over_video_opaque_and_transparent():
    # alpha 0 = fully opaque box -> box color wins
    assert caption.composite_box_over_video((10, 20, 30), 0, (200, 200, 200)) == (10, 20, 30)
    # alpha 255 = fully transparent box -> video shows through
    assert caption.composite_box_over_video((10, 20, 30), 255, (200, 200, 200)) == (200, 200, 200)


# ---------------------------------------------------------------------------
# Time formatting
# ---------------------------------------------------------------------------

def test_sec_to_ass():
    assert caption.sec_to_ass(0) == "0:00:00.00"
    assert caption.sec_to_ass(3661.5) == "1:01:01.50"
    assert caption.sec_to_ass(1.234) == "0:00:01.23"


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------

def _word(w, start, end):
    return {"word": w, "start": start, "end": end}


def test_chunk_words_groups_and_bounds():
    words = [_word(str(i), i, i + 1) for i in range(5)]
    chunks = caption.chunk_words(words, 2)
    assert [len(c["words"]) for c in chunks] == [2, 2, 1]
    assert chunks[0]["start"] == 0 and chunks[0]["end"] == 2
    assert chunks[-1]["start"] == 4 and chunks[-1]["end"] == 5


# ---------------------------------------------------------------------------
# .wts parsing (compound merge, punctuation merge, dedup)
# ---------------------------------------------------------------------------

# Mirrors the real whisper.cpp .wts drawtext format: the active word sits before a
# "|", prefixed with "> " and surrounded by backslash-space padding tokens.
WTS_LINE = (
    "drawtext=fontfile='/f.ttf':fontsize=24:fontcolor=lightgreen:"
    "x=(w-text_w)/2+8:y=h/2:"
    r"text='> \ \ {word}|\ \ \ ':"
    "enable='between(t,{start},{end})'"
)


def _wts(tokens):
    return ",".join(
        WTS_LINE.format(word=w, start=f"{s:.2f}", end=f"{e:.2f}") for w, s, e in tokens
    )


def test_parse_wts_compound_and_punct_and_dedup(tmp_path):
    tokens = [
        ("Cloud", 0.00, 0.10),
        ("fl", 0.10, 0.20),
        ("are", 0.20, 0.30),
        ("are", 0.20, 0.30),   # duplicate (word, start) -> deduped before merge
        ("world", 0.30, 0.50),
        (",", 0.50, 0.55),
    ]
    path = tmp_path / "t.wts"
    path.write_text(_wts(tokens), encoding="utf-8")

    words = caption.parse_wts(str(path))
    assert [w["word"] for w in words] == ["Cloudflare", "world,"]
    assert words[0]["start"] == 0.0 and words[0]["end"] == 0.30
    assert words[1]["end"] == 0.55


# ---------------------------------------------------------------------------
# Color selection
# ---------------------------------------------------------------------------

def test_select_colors_empty_palette_falls_back():
    colors = caption.select_colors([], [])
    assert colors["fallback"] is True
    assert colors["text_rgb"] == (255, 255, 255)
    assert colors["box_rgb"] == (0, 0, 0)
    for key in ("text", "box", "box_alpha", "contrast"):
        assert key in colors


def test_select_colors_returns_accessible_pair():
    palette = [(20, 40, 120), (200, 180, 60), (240, 240, 240), (10, 10, 10)]
    pixels = palette * 8
    colors = caption.select_colors(palette, pixels)
    assert set(colors) >= {"text", "box", "text_rgb", "box_rgb", "box_alpha", "contrast"}
    assert colors["contrast"] >= caption.MIN_CONTRAST


# ---------------------------------------------------------------------------
# ASS generation
# ---------------------------------------------------------------------------

def _dialogue_lines(ass):
    return [ln for ln in ass.splitlines() if ln.startswith("Dialogue:")]


def _chunk(colors=None):
    chunk = {
        "words": [_word("Hello", 0.0, 0.5), _word("world", 0.5, 1.0)],
        "start": 0.0,
        "end": 1.0,
    }
    if colors is not None:
        chunk["colors"] = colors
    return chunk


def test_build_ass_plain_has_no_inline_color_overrides():
    ass = caption.build_ass([_chunk()])
    box, caption_line = _dialogue_lines(ass)
    assert ",Box," in box and r"\1c" not in box
    assert ",Caption," in caption_line and r"\1c" not in caption_line


def test_build_ass_per_chunk_injects_color_overrides():
    colors = {
        "text_rgb": (245, 233, 200),
        "box_rgb": (26, 20, 11),
        "box_alpha": 0x48,
        "contrast": 9.3,
        "fallback": False,
    }
    ass = caption.build_ass([_chunk(colors)])
    box, caption_line = _dialogue_lines(ass)
    assert r"\1c&H0B141A&" in box and r"\1a&H48&" in box
    assert r"\1c&HC8E9F5&" in caption_line


def test_build_ass_global_colors_go_into_header():
    ass = caption.build_ass([_chunk()], colors={"text": "&H00ABCDEF", "box": "&H11223344"})
    assert "&H00ABCDEF" in ass
    assert "&H11223344" in ass


# ---------------------------------------------------------------------------
# Console formatting
# ---------------------------------------------------------------------------

def test_format_colors_string():
    colors = {
        "text_rgb": (245, 233, 200),
        "box_rgb": (26, 20, 11),
        "box_alpha": 0x48,
        "contrast": 9.3,
        "fallback": False,
    }
    out = caption.format_colors(colors)
    assert "text #F5E9C8" in out
    assert "box #1A140B" in out
    assert "72% opaque" in out
    assert "contrast 9.3:1" in out


def test_format_colors_marks_fallback():
    colors = {
        "text_rgb": (255, 255, 255),
        "box_rgb": (0, 0, 0),
        "box_alpha": 0x28,
        "contrast": 21.0,
        "fallback": True,
    }
    assert caption.format_colors(colors).endswith("fallback")


# ---------------------------------------------------------------------------
# Prompt handling
# ---------------------------------------------------------------------------

def test_combine_prompts_joins_non_empty():
    assert caption.combine_prompts("a", "b") == "a\n\nb"
    assert caption.combine_prompts("a", None) == "a"
    assert caption.combine_prompts(None, "b") == "b"
    assert caption.combine_prompts("  ", "") == ""


def test_read_transcript_prompt(tmp_path):
    p = tmp_path / "script.txt"
    p.write_text("  hello world  \n", encoding="utf-8")
    assert caption.read_transcript_prompt(str(p)) == "hello world"


def test_read_transcript_prompt_missing_raises():
    with pytest.raises(ValueError):
        caption.read_transcript_prompt("/no/such/file.txt")


# ---------------------------------------------------------------------------
# Misc helpers
# ---------------------------------------------------------------------------

def test_model_short_name():
    assert caption._model_short_name("/x/ggml-medium.en.bin") == "medium.en"
    assert caption._model_short_name("ggml-base.en.bin") == "base.en"


def test_rounded_rect_is_a_drawing_command():
    d = caption.rounded_rect(0, 0, 100, 50, 10)
    assert d.startswith("m ")   # move-to
    assert " b " in d           # bezier curves for the corners
    assert " l " in d           # line segments for the edges


def test_get_caption_layout_alignment_per_position():
    assert caption.get_caption_layout("top", 1920, 1080)[0] == "8"
    assert caption.get_caption_layout("center", 1920, 1080)[0] == "5"
    assert caption.get_caption_layout("bottom", 1920, 1080)[0] == "2"


def test_get_caption_layout_rejects_bad_position():
    with pytest.raises(ValueError):
        caption.get_caption_layout("sideways", 1920, 1080)


def test_min_contrast_over_samples_no_pixels_uses_box_color():
    # With no sampled pixels it should fall back to plain text-vs-box contrast.
    assert caption.min_contrast_over_samples((255, 255, 255), (0, 0, 0), 0, []) == 21.0
