from __future__ import annotations

from cxreason.controller.controller import LocalRepairController
from cxreason.pipelines.tasks import build_task_controller


def build_cardiomegaly_controller(max_attempts: int = 3) -> LocalRepairController:
    return build_task_controller("cardiomegaly", max_attempts=max_attempts)
