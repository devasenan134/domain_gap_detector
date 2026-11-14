"""Score an image folder against a real reference: python gap_score.py IMAGES."""

import argparse
import json

from src.gap_scorer.runtime import add_embedding_arguments, load_batches
from src.gap_scorer.score import distances


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_embedding_arguments(parser)
    args = parser.parse_args()
    reference, incoming, metadata = load_batches(args)
    print(json.dumps({**metadata, **distances(incoming, reference),
                      "note": "DINOv2 feature distances; lower is closer, not a predicted mAP"},
                     indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
