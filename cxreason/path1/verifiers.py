"""Native Path-1 verifiers kept separate from final metric evaluation."""

from __future__ import annotations

from typing import Any

from cxreason.data.cxreasonbench import NativeStageTurn
from cxreason.evaluation.choice_scorer import score_choice
from cxreason.gates.base import GateResult


class OracleAnswerVerifier:
    """Exact reference-answer verifier for upper-bound experiments only."""

    name = "oracle_answer"

    def verify(
        self,
        turn: NativeStageTurn,
        response: str,
        accepted_responses: dict[str, str],
    ) -> GateResult:
        del accepted_responses
        result = score_choice(
            stage=turn.scorer_stage,
            question=turn.question,
            answer=turn.answer,
            response=response,
        )
        return GateResult(
            passed=result.score == 1,
            score=float(result.score),
            reason=None if result.score == 1 else "stage_not_verified",
            metadata={
                "verification_level": "oracle",
                "verifier": self.name,
                "selected": list(result.selected),
            },
        )

