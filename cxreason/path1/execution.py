"""Stage-selective execution and local repair for native Path 1."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

from cxreason.data.cxreasonbench import NativePath1Case, NativeStageTurn
from cxreason.evaluation.choice_scorer import ChoiceScore, score_choice
from cxreason.gates.base import GateResult
from cxreason.modeling.medgemma_session import HistoryEntry
from cxreason.path1.graph import NativePath1Graph, normalize_repair_stages


@dataclass(frozen=True)
class NativeGeneration:
    text: str
    generated_tokens: int = 0


class StageGenerator(Protocol):
    def generate(
        self,
        turn: NativeStageTurn,
        accepted_history: list[HistoryEntry],
        repair_feedback: str | None = None,
    ) -> NativeGeneration | str:
        ...


class NativeStageVerifier(Protocol):
    name: str

    def verify(
        self,
        turn: NativeStageTurn,
        response: str,
        accepted_responses: dict[str, str],
    ) -> GateResult:
        ...


class AcceptedHistoryFormatter(Protocol):
    """Optionally change an accepted response only in subsequent model history."""

    def __call__(
        self,
        turn: NativeStageTurn,
        response: str,
        attempt_index: int,
        repair_enabled: bool,
    ) -> str:
        ...


@dataclass(frozen=True)
class NativeStageAttempt:
    stage_key: str
    stage_group: str
    attempt_index: int
    response: str
    evaluator: ChoiceScore
    gate_result: GateResult
    repair_enabled: bool
    verifier_name: str | None
    generated_tokens: int


@dataclass
class NativePath1ExecutionState:
    case_id: str
    task: str
    repair_stages: frozenset[str]
    attempts: list[NativeStageAttempt] = field(default_factory=list)
    accepted_responses: dict[str, str] = field(default_factory=dict)
    accepted_turns: list[tuple[NativeStageTurn, str]] = field(default_factory=list)
    model_calls: int = 0
    verifier_calls: int = 0
    evaluator_calls: int = 0
    generated_tokens: int = 0
    failed_stage_key: str | None = None
    failed_stage_group: str | None = None
    stop_reason: str | None = None
    completed_full_path: bool = False
    completed_requested_scope: bool = False

    @property
    def passed(self) -> bool:
        return self.failed_stage_key is None and (
            self.completed_full_path or self.completed_requested_scope
        )

    @property
    def accepted_history(self) -> list[HistoryEntry]:
        return [
            (turn.question, list(turn.image_paths), response)
            for turn, response in self.accepted_turns
        ]

    def attempts_for(self, stage_key: str) -> tuple[NativeStageAttempt, ...]:
        return tuple(attempt for attempt in self.attempts if attempt.stage_key == stage_key)

    def group_summary(self) -> dict[str, dict[str, int]]:
        summary: dict[str, dict[str, int]] = {}
        for attempt in self.attempts:
            values = summary.setdefault(
                attempt.stage_group,
                {"attempts": 0, "repairs": 0, "accepted_turns": 0},
            )
            values["attempts"] += 1
            if attempt.attempt_index > 1:
                values["repairs"] += 1
        for turn, _ in self.accepted_turns:
            values = summary.setdefault(
                turn.scorer_stage,
                {"attempts": 0, "repairs": 0, "accepted_turns": 0},
            )
            values["accepted_turns"] += 1
        return summary


_SAFE_FEEDBACK = {
    "criteria": (
        "The diagnostic criterion in the previous response could not be verified. "
        "Reconsider only this criterion question and select the best option."
    ),
    "custom_criteria": (
        "The criterion-refinement response could not be verified. "
        "Reconsider only this question and select the best option."
    ),
    "bodypart": (
        "The anatomical selection in the previous response could not be verified. "
        "Re-examine only the displayed candidates and select the best option or options."
    ),
    "measurement": (
        "The measurement or recognition response could not be verified. "
        "Reconsider only this stage using the accepted prior context."
    ),
    "final": (
        "The final decision could not be verified. "
        "Reconsider only the decision using the accepted prior context."
    ),
}


def safe_repair_feedback(stage_group: str) -> str:
    """Return stage-specific feedback containing no verifier diagnostics."""

    try:
        return _SAFE_FEEDBACK[stage_group]
    except KeyError as exc:
        raise ValueError(f"No safe repair feedback for {stage_group!r}") from exc


class NativePath1Executor:
    """Run the dynamic graph with independently selectable repair stages."""

    def __init__(
        self,
        *,
        repair_stages: tuple[str, ...] | list[str] | frozenset[str] = (),
        verifiers: dict[str, NativeStageVerifier] | None = None,
        max_attempts_per_stage: int = 3,
        max_attempts_by_stage: dict[str, int] | None = None,
        repair_feedback_overrides: dict[str, str] | None = None,
        repair_feedback_schedules: dict[str, tuple[str, ...] | list[str]] | None = None,
        accepted_history_formatter: AcceptedHistoryFormatter | None = None,
    ) -> None:
        if max_attempts_per_stage <= 0:
            raise ValueError("max_attempts_per_stage must be positive")
        self.repair_stages = normalize_repair_stages(repair_stages)
        self.verifiers = dict(verifiers or {})
        missing = self.repair_stages - set(self.verifiers)
        if missing:
            raise ValueError(f"Missing verifiers for repair stages: {sorted(missing)}")
        self.max_attempts_per_stage = max_attempts_per_stage
        self.max_attempts_by_stage: dict[str, int] = {}
        for label, attempts in (max_attempts_by_stage or {}).items():
            group = next(iter(normalize_repair_stages([label])))
            if group in self.max_attempts_by_stage:
                raise ValueError(f"Duplicate per-stage attempt budget for {group!r}")
            if isinstance(attempts, bool) or not isinstance(attempts, int) or attempts <= 0:
                raise ValueError("Per-stage attempt budgets must be positive integers")
            self.max_attempts_by_stage[group] = attempts
        unused_budgets = set(self.max_attempts_by_stage) - self.repair_stages
        if unused_budgets:
            raise ValueError(
                "Per-stage attempt budgets require the corresponding repair stage: "
                f"{sorted(unused_budgets)}"
            )
        self.repair_feedback_overrides = dict(repair_feedback_overrides or {})
        unknown_feedback_groups = set(self.repair_feedback_overrides) - set(
            _SAFE_FEEDBACK
        )
        if unknown_feedback_groups:
            raise ValueError(
                "Unknown repair-feedback stages: "
                f"{sorted(unknown_feedback_groups)}"
            )
        if any(not value.strip() for value in self.repair_feedback_overrides.values()):
            raise ValueError("Repair-feedback overrides must be non-empty")
        raw_schedules = repair_feedback_schedules or {}
        self.repair_feedback_schedules = {
            group: tuple(values) for group, values in raw_schedules.items()
        }
        unknown_schedule_groups = set(self.repair_feedback_schedules) - set(
            _SAFE_FEEDBACK
        )
        if unknown_schedule_groups:
            raise ValueError(
                "Unknown repair-feedback schedule stages: "
                f"{sorted(unknown_schedule_groups)}"
            )
        if set(self.repair_feedback_overrides) & set(self.repair_feedback_schedules):
            raise ValueError(
                "A repair stage cannot use both a feedback override and a schedule"
            )
        if any(
            not values or any(not isinstance(value, str) or not value.strip() for value in values)
            for values in self.repair_feedback_schedules.values()
        ):
            raise ValueError("Repair-feedback schedules must contain non-empty strings")
        if accepted_history_formatter is not None and not callable(
            accepted_history_formatter
        ):
            raise TypeError("accepted_history_formatter must be callable")
        self.accepted_history_formatter = accepted_history_formatter

    @staticmethod
    def _generation(value: NativeGeneration | str) -> NativeGeneration:
        if isinstance(value, NativeGeneration):
            return value
        if isinstance(value, str):
            return NativeGeneration(value)
        raise TypeError("Stage generator must return str or NativeGeneration")

    @staticmethod
    def _evaluate(turn: NativeStageTurn, response: str) -> ChoiceScore:
        return score_choice(
            stage=turn.scorer_stage,
            question=turn.question,
            answer=turn.answer,
            response=response,
        )

    def _record(
        self,
        state: NativePath1ExecutionState,
        *,
        turn: NativeStageTurn,
        attempt_index: int,
        generation: NativeGeneration,
        evaluator: ChoiceScore,
        gate_result: GateResult,
        repair_enabled: bool,
        verifier_name: str | None,
    ) -> None:
        state.attempts.append(
            NativeStageAttempt(
                stage_key=turn.key,
                stage_group=turn.scorer_stage,
                attempt_index=attempt_index,
                response=generation.text,
                evaluator=evaluator,
                gate_result=gate_result,
                repair_enabled=repair_enabled,
                verifier_name=verifier_name,
                generated_tokens=generation.generated_tokens,
            )
        )
        state.model_calls += 1
        state.evaluator_calls += 1
        state.generated_tokens += generation.generated_tokens

    def _accept(
        self,
        state: NativePath1ExecutionState,
        turn: NativeStageTurn,
        response: str,
        *,
        attempt_index: int,
        repair_enabled: bool,
    ) -> None:
        history_response = response
        if self.accepted_history_formatter is not None:
            history_response = self.accepted_history_formatter(
                turn, response, attempt_index, repair_enabled
            )
            if not isinstance(history_response, str) or not history_response.strip():
                raise ValueError(
                    "accepted_history_formatter must return a non-empty string"
                )
        state.accepted_responses[turn.key] = response
        state.accepted_turns.append((turn, history_response))

    def run(
        self,
        case: NativePath1Case,
        generator: StageGenerator,
        *,
        stop_after_stage_key: str | None = None,
    ) -> NativePath1ExecutionState:
        graph = NativePath1Graph.from_case(case)
        if stop_after_stage_key is not None:
            graph.node(stop_after_stage_key)
        state = NativePath1ExecutionState(
            case_id=case.dicom,
            task=case.task,
            repair_stages=self.repair_stages,
        )

        for node in graph.nodes:
            turn = node.turn
            if not node.repairable:
                generation = self._generation(
                    generator.generate(turn, state.accepted_history, None)
                )
                evaluator = self._evaluate(turn, generation.text)
                entered_path1 = evaluator.score != -1
                gate_result = GateResult(
                    passed=entered_path1,
                    score=float(evaluator.score),
                    reason=None if entered_path1 else "initial_idk",
                    metadata={"control": "path1_entry"},
                )
                self._record(
                    state,
                    turn=turn,
                    attempt_index=1,
                    generation=generation,
                    evaluator=evaluator,
                    gate_result=gate_result,
                    repair_enabled=False,
                    verifier_name=None,
                )
                if not entered_path1:
                    state.failed_stage_key = turn.key
                    state.failed_stage_group = turn.scorer_stage
                    state.stop_reason = "initial_idk"
                    break
                self._accept(
                    state,
                    turn,
                    generation.text,
                    attempt_index=1,
                    repair_enabled=False,
                )
                if turn.key == stop_after_stage_key:
                    state.completed_requested_scope = True
                    return state
                continue

            repair_enabled = node.stage_group in self.repair_stages
            verifier = self.verifiers.get(node.stage_group) if repair_enabled else None
            allowed_attempts = (
                self.max_attempts_by_stage.get(
                    node.stage_group, self.max_attempts_per_stage
                )
                if repair_enabled
                else 1
            )
            accepted = False
            repair_feedback: str | None = None
            for attempt_index in range(1, allowed_attempts + 1):
                generation = self._generation(
                    generator.generate(
                        turn,
                        state.accepted_history,
                        repair_feedback,
                    )
                )
                evaluator = self._evaluate(turn, generation.text)
                if verifier is None:
                    gate_result = GateResult(
                        passed=evaluator.score == 1,
                        score=float(evaluator.score),
                        reason=None if evaluator.score == 1 else "stage_not_verified",
                        metadata={"control": "reference_routing"},
                    )
                    verifier_name = None
                else:
                    gate_result = verifier.verify(
                        turn, generation.text, dict(state.accepted_responses)
                    )
                    state.verifier_calls += 1
                    verifier_name = verifier.name
                self._record(
                    state,
                    turn=turn,
                    attempt_index=attempt_index,
                    generation=generation,
                    evaluator=evaluator,
                    gate_result=gate_result,
                    repair_enabled=repair_enabled,
                    verifier_name=verifier_name,
                )
                if gate_result.passed:
                    self._accept(
                        state,
                        turn,
                        generation.text,
                        attempt_index=attempt_index,
                        repair_enabled=repair_enabled,
                    )
                    accepted = True
                    break
                if attempt_index < allowed_attempts:
                    schedule = self.repair_feedback_schedules.get(node.stage_group)
                    if schedule is not None:
                        repair_feedback = schedule[min(attempt_index - 1, len(schedule) - 1)]
                    else:
                        repair_feedback = self.repair_feedback_overrides.get(
                            node.stage_group, safe_repair_feedback(node.stage_group)
                        )

            if not accepted:
                state.failed_stage_key = turn.key
                state.failed_stage_group = node.stage_group
                state.stop_reason = "repair_budget_exhausted" if repair_enabled else "native_stage_failed"
                break

            if turn.key == stop_after_stage_key:
                state.completed_requested_scope = True
                return state

        if state.failed_stage_key is None:
            state.completed_full_path = True
        return state
