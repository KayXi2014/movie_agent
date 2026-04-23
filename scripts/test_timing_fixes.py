#!/usr/bin/env python3
"""
Quick test to verify timing fixes for >18s cases.
Runs a few test prompts and measures runtime breakdown.
"""

import csv
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[0]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import llm


TEST_PROMPTS = [
    ("Korean thriller with great plot twists", []),
    ("recent horror without gore", []),
    ("French drama about relationships", ["Amélie"]),
    ("90s action movie with good cast", []),
    ("indie film with unique style", []),
]


def measure_recommendation(prompt: str, history: list[Any]) -> dict[str, Any]:
    """Run recommendation and capture timing metrics."""
    started = time.perf_counter()
    
    try:
        result = llm.get_recommendation(prompt, history)
        total_elapsed = time.perf_counter() - started
        
        return {
            "prompt": prompt,
            "history_count": len(history),
            "tmdb_id": result.get("tmdb_id"),
            "used_llm": result.get("used_llm", False),
            "total_s": round(total_elapsed, 3),
            "status": "✓ SUCCESS" if total_elapsed < 18 else "✗ TIMEOUT RISK",
            "description_preview": result.get("description", "")[:60] + "...",
        }
    except Exception as exc:
        total_elapsed = time.perf_counter() - started
        return {
            "prompt": prompt,
            "history_count": len(history),
            "tmdb_id": None,
            "used_llm": False,
            "total_s": round(total_elapsed, 3),
            "status": f"✗ ERROR: {str(exc)[:30]}",
            "description_preview": "",
        }


def run_timing_tests() -> None:
    """Run test suite and report results."""
    print("=" * 100)
    print("TIMING TEST: 18s Runtime Fixes")
    print("=" * 100)
    print()
    
    results = []
    total_time = 0
    over_18s_count = 0
    success_count = 0
    
    for i, (prompt, history) in enumerate(TEST_PROMPTS, start=1):
        print(f"Test {i}/{len(TEST_PROMPTS)}: {prompt[:50]}...")
        
        result = measure_recommendation(prompt, history)
        results.append(result)
        
        total_time += result["total_s"]
        if result["total_s"] >= 18.0:
            over_18s_count += 1
        if result["total_s"] < 18.0:
            success_count += 1
            
        status = result["status"]
        timing = f"{result['total_s']}s"
        llm_used = "LLM" if result["used_llm"] else "Fallback"
        
        print(f"  → {status} | {timing} | {llm_used}")
        print()
    
    # Summary
    print("=" * 100)
    print("RESULTS SUMMARY")
    print("=" * 100)
    print()
    print(f"Tests run:          {len(TEST_PROMPTS)}")
    print(f"Success (<18s):     {success_count}/{len(TEST_PROMPTS)} ({100*success_count//len(TEST_PROMPTS)}%)")
    print(f"Timeout risk (≥18s): {over_18s_count}/{len(TEST_PROMPTS)} ({100*over_18s_count//len(TEST_PROMPTS)}%)")
    print(f"Total time:         {total_time:.1f}s")
    print(f"Average per test:   {total_time/len(TEST_PROMPTS):.1f}s")
    print()
    
    # Detailed table
    print("DETAILED RESULTS")
    print("-" * 100)
    print(f"{'#':<2} {'Prompt':<30} {'Time':<8} {'LLM':<8} {'Status':<20}")
    print("-" * 100)
    
    for i, result in enumerate(results, start=1):
        prompt_short = result["prompt"][:28]
        timing = f"{result['total_s']}s"
        llm_str = "✓" if result["used_llm"] else "✗"
        status = "OK" if result["total_s"] < 18 else "SLOW"
        print(f"{i:<2} {prompt_short:<30} {timing:<8} {llm_str:<8} {status:<20}")
    
    print()
    print("=" * 100)
    
    # Recommendation
    print()
    if over_18s_count == 0:
        print("✓ EXCELLENT: No >18s cases detected!")
        print("  The timing fixes are working well.")
    elif over_18s_count <= 1:
        print("✓ GOOD: Minimal >18s cases (acceptable)")
        print("  The timing fixes are helping.")
    else:
        print(f"⚠ CAUTION: {over_18s_count} cases exceeded 18s")
        print("  May need additional optimization.")
    
    print()


if __name__ == "__main__":
    run_timing_tests()
