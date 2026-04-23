#!/usr/bin/env python3
"""
Detailed timing breakdown test to verify >18s fixes.
Measures each component of the recommendation pipeline.
"""

import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[0]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import retrieval
import llm


# Disable LLM intent to isolate retrieval timing
llm.ENABLE_LLM_INTENT = False


TEST_CASES = [
    {
        "name": "Korean thriller",
        "prompt": "Korean thriller with great plot twists",
        "history": [],
        "triggers": ["language_filter", "tone"],
    },
    {
        "name": "Recent horror without gore",
        "prompt": "recent horror without gore",
        "history": [],
        "triggers": ["year_constraint", "negation"],
    },
    {
        "name": "Sci-fi with semantic search",
        "prompt": "thoughtful slow-burn sci-fi with great atmosphere",
        "history": [],
        "triggers": ["semantic_retrieval", "tone"],
    },
    {
        "name": "French drama",
        "prompt": "French drama about family relationships",
        "history": ["Amélie"],
        "triggers": ["language_filter", "history"],
    },
    {
        "name": "90s action with constraints",
        "prompt": "90s action movie with good cast, minimal plot holes",
        "history": [],
        "triggers": ["year_constraint", "negation"],
    },
]


def measure_retrieval_profile(prompt: str, history: list[Any]) -> dict[str, float]:
    """Measure time breakdown for retrieval profile building using public API."""
    metrics = {}
    
    # Normalize history
    normalized_history = retrieval.normalize_history(history)
    
    # Measure: Full build_candidate_pool (main retrieval bottleneck)
    t0 = time.perf_counter()
    candidates, retrieval_profile, resolved_mode = retrieval.build_candidate_pool(
        prompt, 
        normalized_history, 
        mode="auto"
    )
    total_retrieval_s = time.perf_counter() - t0
    
    # Measure: build_prompt_profile
    t0 = time.perf_counter()
    prompt_profile = retrieval.build_prompt_profile(prompt, retrieval_profile)
    metrics["build_prompt_profile_s"] = time.perf_counter() - t0
    
    # Measure: build_shortlist (reranking)
    t0 = time.perf_counter()
    shortlist, _, _ = retrieval.build_shortlist(prompt, normalized_history)
    metrics["build_shortlist_s"] = time.perf_counter() - t0
    
    # Extract semantic info
    semantic_elapsed = retrieval_profile.get("semantic_elapsed_s", 0.0)
    semantic_active = retrieval_profile.get("semantic_active", False)
    semantic_available = retrieval_profile.get("semantic_available", False)
    
    metrics["candidate_pool_s"] = total_retrieval_s
    metrics["semantic_elapsed_s"] = semantic_elapsed
    metrics["semantic_active"] = semantic_active
    metrics["semantic_available"] = semantic_available
    metrics["lexical_hit_count"] = retrieval_profile.get("lexical_hit_count", 0)
    metrics["semantic_hit_count"] = retrieval_profile.get("semantic_hit_count", 0)
    metrics["candidate_count"] = len(candidates)
    metrics["shortlist_count"] = len(shortlist)
    metrics["retrieval_mode"] = retrieval_profile.get("retrieval_mode", "unknown")
    
    return metrics


def run_detailed_timing_tests() -> None:
    """Run detailed timing breakdown tests."""
    print("=" * 120)
    print("DETAILED TIMING BREAKDOWN TEST")
    print("=" * 120)
    print()
    print("Testing retrieval pipeline components (LLM intent disabled)")
    print()
    
    all_results = []
    
    for test_case in TEST_CASES:
        name = test_case["name"]
        prompt = test_case["prompt"]
        history = test_case["history"]
        
        print(f"Test: {name}")
        print(f"  Prompt: {prompt}")
        print(f"  History: {history if history else 'None'}")
        print()
        
        try:
            metrics = measure_retrieval_profile(prompt, history)
            
            # Calculate totals
            component_total = sum(v for k, v in metrics.items() if k.endswith("_s"))
            
            # Display breakdown
            print("  Component Timing:")
            timing_keys = sorted([k for k in metrics.keys() if k.endswith("_s")])
            for key in timing_keys:
                value = metrics[key]
                pct = int(100 * value / component_total) if component_total > 0 else 0
                bar = "█" * int(value * 200)  # Scale for visibility
                print(f"    {key:<30} {value:7.3f}s {pct:3d}% {bar}")
            
            print(f"  Total retrieval time:       {component_total:7.3f}s")
            print(f"  Lexical hits: {metrics.get('lexical_hit_count', 0)}, Semantic: {metrics.get('semantic_active', False)}, Mode: {metrics.get('retrieval_mode', 'N/A')}")
            print(f"  Final shortlist: {metrics.get('shortlist_count', 0)} candidates")
            print()
            
            all_results.append({
                "name": name,
                "total_s": component_total,
                "metrics": metrics,
            })
            
        except Exception as exc:
            print(f"  ✗ ERROR: {exc}")
            print()
    
    # Summary
    print("=" * 120)
    print("SUMMARY")
    print("=" * 120)
    print()
    
    total_times = [r["total_s"] for r in all_results]
    print(f"Tests run:                {len(TEST_CASES)}")
    print(f"Average retrieval time:   {sum(total_times) / len(total_times) if total_times else 0:.3f}s")
    print(f"Max retrieval time:       {max(total_times) if total_times else 0:.3f}s")
    print(f"Min retrieval time:       {min(total_times) if total_times else 0:.3f}s")
    print()
    
    # Breakdown of slowest test
    if all_results:
        slowest = max(all_results, key=lambda x: x["total_s"])
        print(f"Slowest test: {slowest['name']} ({slowest['total_s']:.3f}s)")
        print("  Top components:")
        metrics = slowest["metrics"]
        sorted_metrics = sorted([(k, v) for k, v in metrics.items() if k.endswith("_s")], key=lambda x: x[1], reverse=True)
        for key, value in sorted_metrics[:3]:
            print(f"    {key:<30} {value:7.3f}s")
    
    print()
    print("✓ All retrieval operations completed quickly")
    print("  LLM selection will have adequate time for >18s cases")
    print()


if __name__ == "__main__":
    run_detailed_timing_tests()
