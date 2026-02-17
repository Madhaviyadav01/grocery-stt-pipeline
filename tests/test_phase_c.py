"""
Test Phase C - Structured Boundary Extraction

Run with:
    pytest test_phase_c.py
"""

import os
import json
import pytest

from pipeline.phase_C_structured_boundary_extraction.extractor import extract_item_boundaries
from pipeline.phase_C_structured_boundary_extraction.run_phase_C import run_phase_C


# -------------------------
# Unit Tests
# -------------------------

def test_extract_with_and_and_comma():
    text = "Dove soap 100 gram small and 2 packets Maggi, Amul butter 500 gram medium"

    expected = [
        "dove soap 100 gram small",
        "2 packets maggi",
        "amul butter 500 gram medium"
    ]

    result = extract_item_boundaries(text)
    assert result == expected


def test_extract_continuous_speech_style():
    """
    Test realistic STT output (no commas, no 'and')
    """
    text = "dow beauty soap 100 gram small 6 amul butter 500 gram medium 2 vitania biscuits 200 gram small"

    expected = [
        "dow beauty soap 100 gram small",
        "6 amul butter 500 gram medium",
        "2 vitania biscuits 200 gram small"
    ]

    result = extract_item_boundaries(text)
    assert result == expected


def test_empty_input():
    assert extract_item_boundaries("") == []


def test_punctuation_cleanup():
    text = "Apple., Banana! and 2 Mangoes?"
    expected = ["apple", "banana", "2 mangoes"]

    result = extract_item_boundaries(text)
    assert result == expected


# -------------------------
# Integration Test
# -------------------------

def test_run_phase_C_integration(tmp_path):
    """
    Full pipeline test using temporary transcript file
    """

    transcript_content = "Milk 1 liter and Bread 2 packets"
    transcript_file = tmp_path / "test_transcript.txt"
    transcript_file.write_text(transcript_content, encoding="utf-8")

    output_dir = tmp_path / "output"

    items, output_path = run_phase_C(
        str(transcript_file),
        output_folder=str(output_dir)
    )

    expected = [
        "milk 1 liter",
        "bread 2 packets"
    ]

    assert items == expected
    assert os.path.exists(output_path)

    # Verify saved JSON content
    with open(output_path, "r", encoding="utf-8") as f:
        saved_items = json.load(f)

    assert saved_items == expected
