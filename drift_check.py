"""Check a held-out image batch for drift: python drift_check.py IMAGES --reference REAL."""

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from src.gap_scorer.drift import drift_test
from src.gap_scorer.runtime import add_embedding_arguments, load_batches


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_embedding_arguments(parser)
    parser.add_argument("--log", type=Path, default=Path("runs/drift.jsonl"))
    args = parser.parse_args()
    reference, incoming, metadata = load_batches(args, disjoint=True)
    result = {"timestamp": datetime.now(timezone.utc).isoformat(), **metadata,
              **drift_test(reference, incoming),
              "note": "A batch-level distribution test, not evidence of detector accuracy loss"}
    record = json.dumps(result, allow_nan=False)
    args.log.parent.mkdir(parents=True, exist_ok=True)
    with args.log.open("a") as log:
        log.write(record + "\n")
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
