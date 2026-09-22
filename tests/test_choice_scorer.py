from __future__ import annotations

import unittest

from cxreason.evaluation.choice_scorer import measurement_matches, score_choice


class ChoiceScorerTest(unittest.TestCase):
    def test_scores_explicit_choice_letter(self) -> None:
        result = score_choice(
            stage="init",
            question="Finding? Options: (a) Yes, (b) No, (c) I don't know",
            answer="(b) No",
            response="FINAL ANSWER: (B) No. The heart is not enlarged.",
        )
        self.assertEqual(result.score, 1)
        self.assertEqual(result.selected, ("b",))

    def test_detects_idk(self) -> None:
        result = score_choice(
            stage="init",
            question="Finding? Options: (a) Yes, (b) No, (c) I don't know",
            answer="(b) No",
            response="FINAL ANSWER: (c) I don't know",
        )
        self.assertEqual(result.score, -1)

    def test_requires_exact_multiselect(self) -> None:
        result = score_choice(
            stage="bodypart",
            question="Select images. Options: (a) 1st image, (b) 2nd image, (c) 3rd image, (d) 4th image, (e) None of the above",
            answer="(b) 2nd image, (d) 4th image",
            response="FINAL ANSWER: (b) 2nd image",
        )
        self.assertEqual(result.score, 0)

    def test_parses_grouped_multiselect_letters(self) -> None:
        result = score_choice(
            stage="bodypart",
            question="Select images. Options: (a) 1st image, (b) 2nd image, (c) 3rd image, (d) 4th image, (e) None of the above",
            answer="(a) 1st image, (b) 2nd image, (c) 3rd image, (d) 4th image",
            response="FINAL ANSWER: (a, b, c, and d)",
        )
        self.assertEqual(result.score, 1)
        self.assertEqual(result.selected, ("a", "b", "c", "d"))

    def test_final_accepts_one_of_multiple_reference_options(self) -> None:
        result = score_choice(
            stage="final",
            question="Decision? Options: (a) Yes, (b) No",
            answer="(a) Yes, (b) No",
            response="FINAL ANSWER: (a) Yes",
        )
        self.assertEqual(result.score, 1)

    def test_projection_multiselect_supports_letters_through_j(self) -> None:
        question = (
            "Choose one per side. (a) Right one, (b) Right two, (c) Right three, "
            "(d) Right four, (e) Right five, (f) Left one, (g) Left two, "
            "(h) Left three, (i) Left four, (j) Left five"
        )
        correct = score_choice(
            stage="measurement",
            question=question,
            answer="(b) Right two, (h) Left three",
            response="FINAL ANSWER: (b), (h)",
        )
        wrong_left = score_choice(
            stage="measurement",
            question=question,
            answer="(b) Right two, (h) Left three",
            response="FINAL ANSWER: (b), (g)",
        )
        self.assertEqual(correct.score, 1)
        self.assertEqual(correct.selected, ("b", "h"))
        self.assertEqual(wrong_left.score, 0)

    def test_measurement_consistency(self) -> None:
        self.assertTrue(
            measurement_matches(
                "cardiomegaly", "(b) [0.50 - 0.60]", "VALUE: 0.55"
            )
        )
        self.assertFalse(
            measurement_matches(
                "cardiomegaly", "(b) [0.50 - 0.60]", "VALUE: 0.70"
            )
        )


if __name__ == "__main__":
    unittest.main()
