from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import pyarrow.dataset as ds


SPLIT_PATTERNS = {
    "train": "train-*.parquet",
    "evaluation": "evaluation-*.parquet",
    "test": "test-*.parquet",
}


def analyze_split(data_dir: Path, split: str, pattern: str) -> dict:
    files = sorted(data_dir.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No files matching {pattern!r} in {data_dir}")
    dataset = ds.dataset([str(p) for p in files], format="parquet")

    qtype_counter: Counter[str] = Counter()
    answer_counter_by_qtype: defaultdict[str, Counter[str]] = defaultdict(Counter)
    images_by_qtype: defaultdict[str, set[str]] = defaultdict(set)
    unique_images: set[str] = set()
    total_rows = 0

    for batch in dataset.to_batches(columns=["question_type", "source_id", "answer"]):
        qtypes = batch.column(0).to_pylist()
        source_ids = batch.column(1).to_pylist()
        answers = batch.column(2).to_pylist()

        total_rows += len(qtypes)
        unique_images.update(source_ids)

        for qtype, source_id, answer in zip(qtypes, source_ids, answers):
            qtype_counter[qtype] += 1
            answer_counter_by_qtype[qtype][str(answer)] += 1
            images_by_qtype[qtype].add(source_id)

    return {
        "split": split,
        "files": [p.name for p in files],
        "qa_pairs": total_rows,
        "images": len(unique_images),
        "question_types": dict(sorted(qtype_counter.items())),
        "question_type_image_counts": {
            qtype: len(images_by_qtype[qtype]) for qtype in sorted(images_by_qtype)
        },
        "question_type_top_answers": {
            qtype: answer_counter_by_qtype[qtype].most_common(10)
            for qtype in sorted(answer_counter_by_qtype)
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize Zoo-Bus-VQA parquet shards.")
    parser.add_argument(
        "--data-dir",
        type=Path,
        required=True,
        help="Directory containing parquet shards.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/dataset_summary.json"),
        help="Destination JSON file.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_dir = args.data_dir.expanduser().resolve()
    split_summaries = {
        split: analyze_split(data_dir, split, pattern)
        for split, pattern in SPLIT_PATTERNS.items()
    }

    overall_qtype_counter: Counter[str] = Counter()
    overall_images_by_qtype: defaultdict[str, set[str]] = defaultdict(set)
    overall_answers_by_qtype: defaultdict[str, Counter[str]] = defaultdict(Counter)
    overall_images: set[str] = set()
    total_rows = sum(split_summary["qa_pairs"] for split_summary in split_summaries.values())

    # Re-scan once for exact overall image counts and per-question aggregates.
    all_files = sorted(data_dir.glob("*.parquet"))
    dataset = ds.dataset([str(p) for p in all_files], format="parquet")
    for batch in dataset.to_batches(columns=["question_type", "source_id", "answer"]):
        qtypes = batch.column(0).to_pylist()
        source_ids = batch.column(1).to_pylist()
        answers = batch.column(2).to_pylist()

        overall_images.update(source_ids)

        for qtype, source_id, answer in zip(qtypes, source_ids, answers):
            overall_qtype_counter[qtype] += 1
            overall_images_by_qtype[qtype].add(source_id)
            overall_answers_by_qtype[qtype][str(answer)] += 1

    summary = {
        "overall": {
            "qa_pairs": total_rows,
            "images": len(overall_images),
            "splits": {
                split: {
                    "qa_pairs": split_summary["qa_pairs"],
                    "images": split_summary["images"],
                    "question_type_count": len(split_summary["question_types"]),
                }
                for split, split_summary in split_summaries.items()
            },
            "question_type_count": len(overall_qtype_counter),
        },
        "per_question_type": {},
    }

    for qtype in sorted(overall_qtype_counter):
        split_counts = {
            split: split_summaries[split]["question_types"].get(qtype, 0)
            for split in SPLIT_PATTERNS
        }
        qa_count = overall_qtype_counter[qtype]
        summary["per_question_type"][qtype] = {
            "qa_pairs": qa_count,
            "images": len(overall_images_by_qtype[qtype]),
            "dataset_share": qa_count / total_rows,
            "split_counts": split_counts,
            "top_answers": overall_answers_by_qtype[qtype].most_common(10),
        }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    print(args.output)


if __name__ == "__main__":
    main()
