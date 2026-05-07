"""Build a clean evaluation subset from a chunked reasoning dataset.

The script keeps examples for which all available labels agree. This is useful for
constructing a benign subset before running verifier experiments.

Expected input format
---------------------
The input JSON file should contain a list of dictionaries. Each dictionary must
contain the keys ``label``, ``rule_score``, and ``learning_score``.

Example
-------
python data_processing_new.py \
    --input data/train5K_chunked_256.json \
    --output data/benign_label1.json \
    --max-input 1100 \
    --max-output 1000
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List


REQUIRED_KEYS = ("label", "rule_score", "learning_score")


def load_json(path: str | Path) -> List[Dict[str, Any]]:
    """Load a JSON list from ``path``."""
    with Path(path).open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, list):
        raise ValueError(f"Expected a JSON list in {path!s}.")
    return data


def keep_label_consistent_examples(
    data: Iterable[Dict[str, Any]],
    max_input: int | None = 1100,
    max_output: int | None = 1000,
) -> List[Dict[str, Any]]:
    """Return examples whose three labels agree.

    Parameters
    ----------
    data:
        Iterable of dataset examples.
    max_input:
        Optional number of examples to inspect from the start of ``data``.
    max_output:
        Optional number of accepted examples to keep.
    """
    selected: List[Dict[str, Any]] = []
    inspected = list(data)[:max_input] if max_input is not None else list(data)

    for item in inspected:
        if not all(key in item for key in REQUIRED_KEYS):
            continue
        if item["label"] == item["rule_score"] == item["learning_score"]:
            selected.append(item)
        if max_output is not None and len(selected) >= max_output:
            break

    return selected


def save_json(data: List[Dict[str, Any]], path: str | Path) -> None:
    """Write ``data`` as formatted JSON."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, ensure_ascii=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="data/train5K_chunked_256.json", help="Input JSON dataset path.")
    parser.add_argument("--output", default="data/benign_label1.json", help="Output JSON path.")
    parser.add_argument("--max-input", type=int, default=1100, help="Maximum number of input examples to inspect.")
    parser.add_argument("--max-output", type=int, default=1000, help="Maximum number of examples to write.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data = load_json(args.input)
    filtered = keep_label_consistent_examples(data, max_input=args.max_input, max_output=args.max_output)
    save_json(filtered, args.output)
    print(f"Wrote {len(filtered)} examples to {args.output}")


if __name__ == "__main__":
    main()
