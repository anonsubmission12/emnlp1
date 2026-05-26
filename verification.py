"""
Lightweight human-in-the-loop verification scaffolding.

Creates three checkpoints per turn and writes them to JSONL for
external annotation workflows. Majority vote can be applied later.
"""

from __future__ import annotations

import json
import os
from typing import Dict, List

CHECKPOINT_TYPES = [
    "checkpoint_1_bias_manipulation",
    "checkpoint_2_policy_compliance",
    "checkpoint_3_quality",
]


def log_checkpoints(
    conversation_id: int,
    turn_no: int,
    user_msg: str,
    agent_msg: str,
    expert_thinking: Dict[str, str],
    output_dir: str = "verification",
) -> List[Dict]:
    """Create checkpoint records and append them to a JSONL log."""
    os.makedirs(output_dir, exist_ok=True)
    records = []

    for ctype in CHECKPOINT_TYPES:
        record = {
            "conversation_id": conversation_id,
            "turn_no": turn_no,
            "checkpoint_type": ctype,
            "user_msg": user_msg,
            "agent_msg": agent_msg,
            "expert_thinking": expert_thinking,
            "status": "pending",
        }
        records.append(record)

    path = os.path.join(output_dir, "checkpoints.jsonl")
    with open(path, "a", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")

    return records


def apply_majority_vote(annotations: List[Dict]) -> Dict:
    """
    Apply majority vote for a single checkpoint.
    Expects annotations with fields: verdict ("accept"|"reject").
    """
    votes = {"accept": 0, "reject": 0}
    for ann in annotations:
        verdict = ann.get("verdict")
        if verdict in votes:
            votes[verdict] += 1

    if votes["accept"] == votes["reject"]:
        return {"verdict": "tie", "votes": votes}

    verdict = "accept" if votes["accept"] > votes["reject"] else "reject"
    return {"verdict": verdict, "votes": votes}
