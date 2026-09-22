"""Dynamic native Path-1 graph, including optional and multi-round stages."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from cxreason.data.cxreasonbench import NativePath1Case, NativeStageTurn


REPAIRABLE_STAGE_GROUPS = (
    "criteria",
    "custom_criteria",
    "bodypart",
    "measurement",
    "final",
)

STAGE_ALIASES = {
    "stage1": "criteria",
    "stage_1": "criteria",
    "criterion": "criteria",
    "criteria": "criteria",
    "stage1.5": "custom_criteria",
    "stage_1.5": "custom_criteria",
    "stage15": "custom_criteria",
    "criterion_refinement": "custom_criteria",
    "custom_criteria": "custom_criteria",
    "stage2": "bodypart",
    "stage_2": "bodypart",
    "anatomy": "bodypart",
    "bodypart": "bodypart",
    "stage3": "measurement",
    "stage_3": "measurement",
    "measurement": "measurement",
    "recognition": "measurement",
    "stage4": "final",
    "stage_4": "final",
    "decision": "final",
    "final": "final",
}


def normalize_repair_stages(stages: Iterable[str]) -> frozenset[str]:
    """Normalize paper stage labels into stable internal group names."""

    raw = [stage.strip().lower() for stage in stages]
    if "all" in raw:
        if len(raw) != 1:
            raise ValueError("'all' cannot be combined with individual repair stages")
        return frozenset(REPAIRABLE_STAGE_GROUPS)
    normalized: set[str] = set()
    for stage in raw:
        if stage in {"init", "initial", "stage0", "stage_0"}:
            raise ValueError("The initial diagnostic decision is fixed and cannot be repaired")
        try:
            normalized.add(STAGE_ALIASES[stage])
        except KeyError as exc:
            raise ValueError(f"Unknown Path-1 repair stage: {stage!r}") from exc
    return frozenset(normalized)


@dataclass(frozen=True)
class NativePath1Node:
    """One node in a case-specific Path-1 execution graph."""

    turn: NativeStageTurn
    stage_group: str
    repairable: bool
    requires_pass_to_continue: bool
    predecessor: str | None
    successor: str | None


@dataclass(frozen=True)
class NativePath1Graph:
    """The complete native graph before model-dependent early termination."""

    task: str
    dicom: str
    nodes: tuple[NativePath1Node, ...]

    @classmethod
    def from_case(cls, case: NativePath1Case) -> "NativePath1Graph":
        turns = list(case.all_turns)
        if not turns or turns[0].key != "init" or turns[-1].key != "final":
            raise ValueError("A Path-1 graph must start at init and end at final")
        keys = [turn.key for turn in turns]
        if len(keys) != len(set(keys)):
            raise ValueError(f"Duplicate Path-1 turn keys for {case.task}/{case.dicom}")
        nodes: list[NativePath1Node] = []
        for index, turn in enumerate(turns):
            is_initial = turn.key == "init"
            nodes.append(
                NativePath1Node(
                    turn=turn,
                    stage_group=turn.scorer_stage,
                    repairable=not is_initial,
                    requires_pass_to_continue=not is_initial,
                    predecessor=turns[index - 1].key if index else None,
                    successor=turns[index + 1].key if index + 1 < len(turns) else None,
                )
            )
        return cls(task=case.task, dicom=case.dicom, nodes=tuple(nodes))

    @property
    def stage_groups(self) -> tuple[str, ...]:
        return tuple(dict.fromkeys(node.stage_group for node in self.nodes))

    def node(self, key: str) -> NativePath1Node:
        for node in self.nodes:
            if node.turn.key == key:
                return node
        raise KeyError(key)

