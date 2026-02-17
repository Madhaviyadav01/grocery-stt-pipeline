"""
main.py — Grocery AI STT Pipeline Orchestrator
================================================
Orchestrates the full end-to-end pipeline by running Phases A through E
sequentially. Each phase is imported from its dedicated module and executed
with structured logging. Execution halts immediately if any phase fails.

Usage:
    python data_pipeline/main.py
"""


import sys
import os
import traceback
from datetime import datetime

from pipeline.phase_A_Audio_preprocessing.run_phase_A import run_phase_A
from pipeline.phase_B_transcription.run_phase_B import run_phase_B
from pipeline.phase_C_structured_boundary_extraction.run_phase_C import run_phase_C
from pipeline.phase_D_fuzzy_canonical_mapping.run_phase_D import run_phase_D
from pipeline.phase_E_evaluation.run_phase_E import run_phase_E



# ---------------------------------------------------------------------------
# Logging helpers
# ---------------------------------------------------------------------------

def _timestamp() -> str:
    """Return a human-readable UTC timestamp."""
    return datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")


def _log_phase_start(phase: str, description: str) -> None:
    print(f"\n{'=' * 60}")
    print(f"  [{_timestamp()}]  STARTING  {phase}")
    print(f"  {description}")
    print(f"{'=' * 60}")


def _log_phase_success(phase: str) -> None:
    print(f"[{_timestamp()}]  ✓  {phase} completed successfully.")


def _log_phase_failure(phase: str, exc: Exception) -> None:
    print(f"\n{'!' * 60}", file=sys.stderr)
    print(f"  [{_timestamp()}]  ✗  {phase} FAILED", file=sys.stderr)
    print(f"  Error: {exc}", file=sys.stderr)
    print(f"{'!' * 60}\n", file=sys.stderr)
    traceback.print_exc()


# ---------------------------------------------------------------------------
# Pipeline definition
# ---------------------------------------------------------------------------

_PHASES = [
    {
        "name": "Phase A — Audio Preprocessing",
        "description": "Normalise, denoise, and segment raw audio files.",
        "runner": run_phase_A,
    },
    {
        "name": "Phase B — Transcription",
        "description": "Convert preprocessed audio segments to raw text via STT.",
        "runner": run_phase_B,
    },
    {
        "name": "Phase C — Structured Boundary Extraction",
        "description": "Detect item boundaries and extract structured entities from transcripts.",
        "runner": run_phase_C,
    },
    {
        "name": "Phase D — Fuzzy Canonical Mapping",
        "description": "Map extracted entities to canonical grocery product names.",
        "runner": run_phase_D,
    },
    {
        "name": "Phase E — Evaluation",
        "description": "Score pipeline outputs against ground-truth labels.",
        "runner": run_phase_E,
    },
]


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def run_full_pipeline(audio_path: str) -> None:
    """
    Execute all pipeline phases (A to E) sequentially.

    Each phase is wrapped in structured logging. If a phase raises an
    exception the error is printed to stderr and execution stops immediately,
    returning a non-zero exit code.
    """
    print("\n" + "#" * 60)
    print("  Grocery AI STT Pipeline — Full Run")
    print(f"  Started : {_timestamp()}")
    print("#" * 60)
    
    data = audio_path  # initial input

    for phase in _PHASES:
        _log_phase_start(phase["name"], phase["description"])
        try:
            data = phase["runner"](data)
            _log_phase_success(phase["name"])
        except Exception as exc:
            _log_phase_failure(phase["name"], exc)
            sys.exit(1)

    print("\n" + "#" * 60)
    print("  ✓  All phases completed successfully.")
    print(f"  Finished: {_timestamp()}")
    print("#" * 60 + "\n")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python main.py <audio_path>")
        sys.exit(1)

    audio_file = sys.argv[1]
    run_full_pipeline(audio_file)

