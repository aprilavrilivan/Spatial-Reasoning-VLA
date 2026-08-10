from __future__ import annotations

import csv
import json
import shutil
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
DROP_KEYS = {
    "adapter_path",
    "best_checkpoint_path",
    "best_model_checkpoint",
    "output_dir",
    "predictions_csv",
    "predictions_json",
}
MODEL_RUNS = {
    "smol": ROOT / "zoo_bus_vqa_finetune/outputs/smol/smol_final_20260502_120924",
    "qwen": ROOT / "zoo_bus_vqa_finetune/outputs/qwen/qwen_final_evalfix_20260503_073400",
    "gemma": ROOT / "zoo_bus_vqa_finetune/outputs/gemma/gemma_final_20260503_214729",
    "internvl": ROOT / "zoo_bus_vqa_finetune/outputs/internvl/internvl_final_20260504_073322",
    "qwen_openspaces": ROOT
    / "zoo_bus_vqa_finetune/outputs/qwen_openspaces/openspaces_qwen_promptfix_relaxed_20260506_095438",
}
MODEL_REPORTS = ("run_summary.json", "best_eval_metrics.json", "final_test_report.json")


def load_json(path: Path) -> Any:
    return json.loads(path.read_text())


def sanitize(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: sanitize(item)
            for key, item in value.items()
            if key not in DROP_KEYS
        }
    if isinstance(value, list):
        return [sanitize(item) for item in value]
    return value


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def curate_model_reports() -> None:
    for model, run_dir in MODEL_RUNS.items():
        for name in MODEL_REPORTS:
            source = run_dir / "reports" / name
            if source.exists():
                write_json(RESULTS / "model_evaluation" / model / name, sanitize(load_json(source)))


def curate_prompt_ablation() -> None:
    source_root = ROOT / "zoo_bus_vqa_finetune/outputs/prompt_ablation"
    selected_names = {
        "summary.json",
        "evaluation_prompt_variants.json",
        "test_prompt_variants.json",
    }
    if not source_root.exists():
        return
    for source in source_root.rglob("*.json"):
        if source.name in selected_names:
            destination = RESULTS / "prompt_ablation" / source.relative_to(source_root)
            write_json(destination, sanitize(load_json(source)))


def curate_external_evaluation() -> None:
    source_root = ROOT / "zoo_bus_vqa_finetune/external_eval"
    sources = sorted(source_root.glob("results/*/*/external_spatial_eval_summary.json"))
    sources += sorted(
        source_root.glob("robotic_spatial_results/*/*/robotic_spatial_eval_summary.json")
    )
    rows: list[dict[str, Any]] = []
    for source in sources:
        report = load_json(source)
        for dataset_name, dataset in report["datasets"].items():
            before = dataset["reports"]["before"]["metrics"]
            after = dataset["reports"]["after"]["metrics"]
            rows.append(
                {
                    "model": report["model_family"],
                    "dataset": dataset_name,
                    "num_samples": dataset["num_samples"],
                    "before_accuracy": before["overall_accuracy"],
                    "after_accuracy": after["overall_accuracy"],
                    "delta_accuracy": dataset["delta_metrics"]["overall_accuracy"],
                }
            )

    destination = RESULTS / "external_evaluation/summary.csv"
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "model",
                "dataset",
                "num_samples",
                "before_accuracy",
                "after_accuracy",
                "delta_accuracy",
            ),
        )
        writer.writeheader()
        writer.writerows(sorted(rows, key=lambda row: (row["model"], row["dataset"])))


def curate_robot_evaluation() -> None:
    source_root = ROOT / "robot_deployment_qwen/dynamic_unit_tests"
    destination_root = RESULTS / "robot_evaluation/dynamic_unit_tests"
    for name in (
        "summary_by_test_20260529.csv",
        "summary_by_test_20260529.json",
        "cleaning_report_20260529.csv",
        "cleaning_report_20260529.json",
        "manual_annotation_corrections_20260529.csv",
        "manual_annotation_corrections_20260529.json",
    ):
        source = source_root / name
        if source.exists():
            destination_root.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination_root / name)

    complex_rows: list[dict[str, Any]] = []
    for source in sorted((ROOT / "robot_deployment_qwen/complex_navigation").glob("dynamic_*/results.json")):
        events = load_json(source)
        final = next(
            (event for event in reversed(events) if event.get("row_type") == "final"),
            None,
        )
        if final is None:
            continue
        complex_rows.append(
            {
                "run_id": source.parent.name,
                "run_label": final.get("run_label"),
                "controller_complete": final.get("model_loop_success"),
                "physical_success": final.get("human_success"),
                "failure_reason": final.get("failure_reason"),
            }
        )

    destination = RESULTS / "robot_evaluation/complex_navigation_summary.csv"
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "run_id",
                "run_label",
                "controller_complete",
                "physical_success",
                "failure_reason",
            ),
        )
        writer.writeheader()
        writer.writerows(complex_rows)


def preserve_dataset_snapshot() -> None:
    source = ROOT / "zoo_bus_vqa_stats.json"
    if source.exists():
        destination = RESULTS / "dataset/pre_rebalance_dataset_stats.json"
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)


def main() -> None:
    curate_model_reports()
    curate_prompt_ablation()
    curate_external_evaluation()
    curate_robot_evaluation()
    preserve_dataset_snapshot()
    print(RESULTS)


if __name__ == "__main__":
    main()
