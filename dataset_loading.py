"""
convert_to_agentharm.py
=======================
Converts the generated ToolSafe benchmark dataset into the AgentHarm
trajectory format expected by AgentHarmProcessor, then writes:

  harmful_steps.json   — all score=1 entries
  benign_steps.json    — all score=0 entries
  all_steps.json       — full combined dataset

Usage:
  python convert_to_agentharm.py \
      --input  benchmark_final.json \
      --outdir ./TS-Bench/agentharm-traj/

New fields passed through unchanged:
  injection_style      "explicit" | "hidden" | "none"
  injection_delay      trigger_turn - injection_turn  (meaningful for hidden)
  conversation_depth   trigger_turn  (depth at which action fires)
  attack_category      e.g. "hidden_delay_4", "long_explicit", "benign_safe"
"""

import json
import re
import argparse
from pathlib import Path
from collections import Counter


# ─────────────────────────────────────────────────────────────────────────────
# Parsing helpers
# ─────────────────────────────────────────────────────────────────────────────

def parse_current_action(raw: str):
    """
    Split a raw current_action string into (action_name, args_dict).

    Handles both formats:
        "[Action] tool_name Action Input: {...}"
        "[Action] tool_name\\nAction Input: {...}"
    """
    body = raw.strip()
    if body.startswith("[Action]"):
        body = body[len("[Action]"):].strip()

    parts = re.split(r"\s*Action Input:\s*", body, maxsplit=1)
    if len(parts) != 2:
        raise ValueError(f"Cannot parse current_action:\n{raw!r}")

    action_name = parts[0].strip()
    args_str    = parts[1].strip()

    try:
        args_dict = json.loads(args_str)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"Cannot JSON-parse Action Input for '{action_name}':\n{args_str!r}"
        ) from exc

    return action_name, args_dict


def extract_injected_preview(instruction: str, history: str) -> str:
    """
    Return the first 120 chars of whichever text (instruction or history)
    contains the most injection-marker patterns.
    """
    MARKERS = [
        r"#\s*\w+_hook",
        r"<!--",
        r"/\*.*\*/",
        r"--\s*\w+:",
        r"\[SYSTEM",
        r"\[DEBUG\]",
        r"observability_relay",
        r"pipeline_meta",
        r"data_relay",
        r"telemetry_hook",
        r"on_completion",
        r"POSTSCRIPT:",
        r"_pipeline",
        r"mirror_to",
        r"provisioning_agent",
        r"platform_init",
        r"rotation_policy",
        r"delivery_copy",
        r"runtime_relay",
        r"dr_sync",
    ]

    best_src, best_score = instruction, 0
    for src in [instruction, history]:
        score = sum(1 for m in MARKERS
                    if re.search(m, src, re.IGNORECASE | re.DOTALL))
        if score > best_score:
            best_score, best_src = score, src

    return " ".join(best_src.split())[:120]


# ─────────────────────────────────────────────────────────────────────────────
# Entry converter
# ─────────────────────────────────────────────────────────────────────────────

# New benchmark fields that are passed through if present
PASSTHROUGH_FIELDS = [
    "injection_style",
    "injection_delay",
    "conversation_depth",
    "attack_category",
]


def convert_entry(entry: dict) -> dict:
    """
    Convert one entry to AgentHarm trajectory format.

    Always-present output fields
    ----------------------------
    instruction              str
    history                  str
    current_action           str   canonical "[Action] tool\\nAction Input: {...}"
    env_info                 str   JSON string
    score                    float 0.0 or 1.0
    attack_type              str
    injection_turn           int
    trigger_turn             int
    injected_content_preview str

    Passed through when present
    ---------------------------
    injection_style      "explicit" | "hidden" | "none"
    injection_delay      int
    conversation_depth   int
    attack_category      str
    """
    action_name, args_dict = parse_current_action(entry["current_action"])

    canonical_action = (
        f"[Action] {action_name}\n"
        f"Action Input: {json.dumps(args_dict)}"
    )

    # Validate and normalise env_info (must be a JSON string)
    env_info_str = json.dumps(json.loads(entry["env_info"]))

    out = {
        "instruction":              entry["instruction"],
        "history":                  entry.get("history", ""),
        "current_action":           canonical_action,
        "env_info":                 env_info_str,
        "score":                    float(entry["score"]),
        "attack_type":              entry["attack_type"],
        "injection_turn":           entry["injection_turn"],
        "trigger_turn":             entry["trigger_turn"],
        "injected_content_preview": extract_injected_preview(
            entry["instruction"], entry.get("history", "")
        ),
    }

    for field in PASSTHROUGH_FIELDS:
        if field in entry:
            out[field] = entry[field]

    return out


# ─────────────────────────────────────────────────────────────────────────────
# Stats printer
# ─────────────────────────────────────────────────────────────────────────────

def print_stats(converted: list) -> None:
    harmful  = [e for e in converted if e["score"] == 1.0]
    benign   = [e for e in converted if e["score"] == 0.0]
    explicit = [e for e in harmful if e.get("injection_style") == "explicit"]
    hidden   = [e for e in harmful if e.get("injection_style") == "hidden"]

    print(f"\nDataset distribution")
    print(f"  Total              : {len(converted)}")
    print(f"  Harmful            : {len(harmful)}")

    print(f"    Explicit ({len(explicit)})")
    if explicit:
        depth_dist = Counter(
            e.get("conversation_depth", e["trigger_turn"]) for e in explicit
        )
        print(f"      conversation_depth : {dict(sorted(depth_dist.items()))}")

    print(f"    Hidden   ({len(hidden)})")
    if hidden:
        delay_dist = Counter(
            e.get("injection_delay",
                  e["trigger_turn"] - e["injection_turn"]) for e in hidden
        )
        print(f"      injection_delay    : {dict(sorted(delay_dist.items()))}")

    clearly   = [e for e in benign if "suspicious" not in e.get("attack_type", "")]
    suspicion = [e for e in benign if "suspicious"     in e.get("attack_type", "")]
    print(f"  Benign             : {len(benign)}")
    print(f"    Clearly benign      : {len(clearly)}")
    print(f"    Suspicious-but-safe : {len(suspicion)}")

    print(f"\n  Tool distribution (harmful):")
    tool_counts = Counter(
        e["current_action"].split("[Action] ")[1].split("\n")[0].strip()
        for e in harmful
    )
    for tool, count in sorted(tool_counts.items(), key=lambda x: -x[1]):
        print(f"    {tool:<20} {count}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def convert_dataset(input_path: str, outdir: str) -> None:
    input_path = Path(input_path)
    outdir     = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    with open(input_path, encoding="utf-8") as f:
        raw_dataset = json.load(f)

    print(f"Loaded {len(raw_dataset)} entries from {input_path}")

    converted, errors = [], []

    for i, entry in enumerate(raw_dataset):
        try:
            converted.append(convert_entry(entry))
        except Exception as exc:
            errors.append((i, entry.get("attack_type", "?"), str(exc)))
            print(f"  ⚠  entry {i} ({entry.get('attack_type','?')}): {exc}")

    if errors:
        print(f"\n{len(errors)} entries failed conversion.")
    else:
        print(f"All {len(converted)} entries converted successfully.")

    print_stats(converted)

    harmful = [e for e in converted if e["score"] == 1.0]
    benign  = [e for e in converted if e["score"] == 0.0]

    outputs = {
        "harmful_steps.json": harmful,
        "benign_steps.json":  benign,
        "all_steps.json":     converted,
    }

    print()
    for filename, data in outputs.items():
        path = outdir / filename
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        print(f"  Wrote {len(data):3d} entries → {path}")

    print("\nDone.")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert ToolSafe benchmark to AgentHarm trajectory format."
    )
    parser.add_argument(
        "--input",
        default="dataset.json",
        help="Path to source dataset JSON (default: dataset.json)",
    )
    parser.add_argument(
        "--outdir",
        default="./TS-Bench/agentharm-traj/",
        help="Output directory (default: ./TS-Bench/agentharm-traj/)",
    )
    args = parser.parse_args()
    convert_dataset(args.input, args.outdir)