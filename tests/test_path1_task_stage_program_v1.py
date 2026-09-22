import json
from pathlib import Path


def test_program_covers_all_tasks_and_stages():
    root = Path(__file__).resolve().parents[1]
    config = json.loads((root / "configs/path1_task_stage_program_v1.json").read_text(encoding="utf-8"))
    assert len(config["tasks"]) == 12
    assert config["stage_order"] == ["stage1", "stage1.5", "stage2", "stage3", "stage4"]
    assert config["release_gates"]["claim_all_tasks_genuinely_supported"] is False
    assert all({"stage1.5", "stage2", "stage3", "stage4"} <= set(row) for row in config["tasks"].values())


def test_program_does_not_overclaim_existing_task_candidates():
    root = Path(__file__).resolve().parents[1]
    config = json.loads((root / "configs/path1_task_stage_program_v1.json").read_text(encoding="utf-8"))
    tasks = config["tasks"]
    assert all(tasks["inclusion"][stage] == "frozen_active" for stage in ("stage2", "stage3", "stage4"))
    assert tasks["inspiration"]["stage3"] == "validated_inactive"
    assert tasks["inspiration"]["candidate_completion"] == 13
    assert tasks["carina_angle"]["stage3"] == "rejected"
    assert "carina_stage3_rejection" in config["evidence"]
    assert tasks["cardiomegaly"]["stage3"] == "validated_inactive"
    assert tasks["cardiomegaly"]["candidate_completion"] == 6
    assert all(tasks["projection"][stage] == "frozen_active" for stage in ("stage2", "stage3", "stage4"))
