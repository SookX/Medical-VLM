"""Conservative Stage-2 ensemble that only rescues selected positive classes."""

from __future__ import annotations

from pathlib import Path

from cxreason.calibration.selective_knn import SelectiveKnnModel
from cxreason.data.cxreasonbench import NativeStageTurn
from cxreason.path1.generic_stage2_verifier import (
    CHESTX_DET_TARGETS,
    ClassifiedCandidate,
    IndependentSelectiveAnatomyVerifier,
    candidate_feature_vector,
)
from cxreason.path1.independent_verifiers import extract_red_overlay_mask


class ComplementarySelectiveAnatomyVerifier(IndependentSelectiveAnatomyVerifier):
    """Use a secondary model only for explicitly locked positive rescue labels."""

    name = "independent_complementary_selective_anatomy_knn_v1"

    def __init__(
        self,
        *,
        task: str,
        prediction,
        primary_policy: dict[str, object],
        primary_model_path: Path,
        rescue_policy: dict[str, object],
        rescue_model_path: Path,
        rescue_labels: tuple[str, ...],
    ) -> None:
        super().__init__(
            task=task,
            prediction=prediction,
            policy=primary_policy,
            model_path=primary_model_path,
        )
        if tuple(rescue_policy["required_types"]) != self.required_types:
            raise ValueError("Rescue policy required-type inventory mismatch")
        unknown = set(rescue_labels) - set(self.required_types)
        if unknown:
            raise ValueError(f"Unknown rescue labels: {sorted(unknown)}")
        self.rescue_model = SelectiveKnnModel(rescue_model_path, rescue_policy)
        self.rescue_labels = frozenset(rescue_labels)

    def _classify_turn(self, turn: NativeStageTurn) -> list[ClassifiedCandidate]:
        output = []
        for index, image_path in enumerate(turn.image_paths):
            letter = chr(ord("a") + index)
            overlay = extract_red_overlay_mask(image_path)
            if overlay is None:
                output.append(ClassifiedCandidate(letter, None, float("inf")))
                continue
            height, width = overlay.shape
            masks = {
                target: self.prediction.project_mask(target, (width, height))
                for target in CHESTX_DET_TARGETS
            }
            features = candidate_feature_vector(overlay, masks)
            label, distance = self.model.predict(features)
            if label is None:
                rescue_label, rescue_distance = self.rescue_model.predict(features)
                if rescue_label in self.rescue_labels:
                    label, distance = rescue_label, rescue_distance
            output.append(ClassifiedCandidate(letter, label, distance))
        return output


class ConsensusRescueSelectiveAnatomyVerifier(IndependentSelectiveAnatomyVerifier):
    """Fill primary abstentions only when two locked rescue models agree."""

    name = "independent_consensus_rescue_selective_anatomy_knn_v1"

    def __init__(
        self,
        *,
        task: str,
        prediction,
        primary_policy: dict[str, object],
        primary_model_path: Path,
        rescue_policy: dict[str, object],
        rescue_model_path: Path,
        confirmation_policy: dict[str, object],
        confirmation_model_path: Path,
    ) -> None:
        super().__init__(
            task=task,
            prediction=prediction,
            policy=primary_policy,
            model_path=primary_model_path,
        )
        for policy in (rescue_policy, confirmation_policy):
            if tuple(policy["required_types"]) != self.required_types:
                raise ValueError("Consensus policy required-type inventory mismatch")
        self.rescue_model = SelectiveKnnModel(rescue_model_path, rescue_policy)
        self.confirmation_model = SelectiveKnnModel(
            confirmation_model_path, confirmation_policy
        )

    def _classify_turn(self, turn: NativeStageTurn) -> list[ClassifiedCandidate]:
        output = []
        for index, image_path in enumerate(turn.image_paths):
            letter = chr(ord("a") + index)
            overlay = extract_red_overlay_mask(image_path)
            if overlay is None:
                output.append(ClassifiedCandidate(letter, None, float("inf")))
                continue
            height, width = overlay.shape
            masks = {
                target: self.prediction.project_mask(target, (width, height))
                for target in CHESTX_DET_TARGETS
            }
            features = candidate_feature_vector(overlay, masks)
            label, distance = self.model.predict(features)
            if label is None:
                rescue_label, rescue_distance = self.rescue_model.predict(features)
                confirmation_label, _ = self.confirmation_model.predict(features)
                if rescue_label is not None and rescue_label == confirmation_label:
                    label, distance = rescue_label, rescue_distance
            output.append(ClassifiedCandidate(letter, label, distance))
        return output
