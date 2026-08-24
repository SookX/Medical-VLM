# Stage-Gated Diagnostic Reasoning with Dependency-Aware Verification for Medical VLMs

## Goal

Build a **training-free inference framework** on top of the public CXReasonBench evaluation pipeline.

The system should keep the medical VLM frozen and modify the standard CXReasonBench **Path 1** evaluation so that intermediate reasoning stages are externally verified.

If a stage fails verification, only that stage is regenerated.

Previously accepted upstream stages are preserved.

The main research question is:

> Can external stage-specific verification and bounded local repair improve CXReasonBench Path-1 reasoning beyond what can be explained by additional inference compute?

---

# Core Research Idea

CXReasonBench already evaluates diagnostic reasoning through multiple intermediate stages.

Normally, these stages are primarily used as evaluation checkpoints.

The proposed system changes them into **active test-time error-control checkpoints**.

The standard process is approximately:

`Generate stage → score stage → continue or fail`

Our process becomes:

`Generate stage → verify stage → PASS or REPAIR → continue`

The VLM remains responsible for the actual reasoning.

External tools do **not** provide the correct answer before generation.

They only determine whether the VLM-generated state is sufficiently supported.

---

# Scope

## Primary Benchmark

Use:

**CXReasonBench Path 1 only**

Do not use Path 2 as part of the primary experiment.

Do not use Re-evaluated Path 1 as part of the primary experiment.

The goal is specifically to improve the model's ability to independently execute the Path-1 reasoning process.

---

# Models

Initial models:

- Qwen2.5-VL-7B
- MedGemma-4B

All model parameters remain frozen.

No:

- SFT
- PPO
- GRPO
- preference optimization
- RL
- model weight updates

This is purely an inference-time intervention.

---

# Main Path-1 Stages

## Stage 1 — Diagnostic Criterion Selection

The model identifies the clinical criterion required for the task.

Examples:

- Cardiomegaly → cardiothoracic ratio
- Carina abnormality → carina-angle criterion
- Mediastinal widening → benchmark-defined mediastinal relation

### Gate

Use a deterministic mapping between:

- diagnostic task
- accepted clinical criterion

The gate returns:

- `PASS`
- `REPAIR`

If the selected criterion is incorrect, ask the model to reconsider the stage without revealing the correct criterion.

---

## Stage 1.5 — Refined Criterion Adoption

Some CXReasonBench tasks contain an expert-defined refined criterion.

Keep this behavior compatible with the original CXReasonBench evaluation.

Do not treat this as a new contribution.

---

## Stage 2 — Anatomical Grounding

The model identifies the anatomical structures or reference lines required for the selected criterion.

Keep the **native CXReasonBench Stage-2 format**.

Do not require new free-form bounding-box generation in the primary experiment.

### Oracle Gate

For the oracle experiment:

- compare the model's Stage-2 answer with the benchmark answer;
- return only `PASS` or `REPAIR`;
- never reveal the correct answer.

### Practical Gate

Where possible, use an independently produced anatomical estimate.

Potential sources:

- HybridGNet
- CheXmask
- another independent segmentation model

Possible comparison methods:

- Dice / IoU for region-based structures
- normalized point distance for landmarks
- normalized line distance for reference lines

### Important Limitation

CheXStruct uses CXAS to generate many of the anatomical structures used by CXReasonBench.

Therefore:

- CXAS is **benchmark-linked**
- CXAS must not be described as an independent verifier

Keep separate concepts for:

1. Oracle/reference verifier
2. Benchmark-linked verifier
3. Independent verifier

---

## Stage 3 — Measurement or Recognition

The model performs the required measurement or qualitative observation.

### Quantitative Example

For cardiomegaly, the state may contain:

- Cardiac width: 55
- Thoracic width: 100
- Cardiothoracic ratio: 0.55

### Gate A — Arithmetic Consistency

Use deterministic code.

Check whether the reported derived quantity matches the component measurements.

Example:

`55 / 100 = 0.55`

If the model reports `0.72`, the gate should fail.

### Gate B — Visual Measurement Agreement

Where an independent anatomical estimate exists:

1. derive the corresponding measurement;
2. map it to the CXReasonBench answer format;
3. compare it with the VLM answer.

### Qualitative Tasks

For categorical tasks, use deterministic or independently grounded checks where feasible.

Avoid adding a generic LLM-as-a-judge unless absolutely necessary.

If a qualitative task cannot be reliably verified, mark that task as unsupported by the practical verifier rather than pretending the verifier is reliable.

---

## Stage 4 — Final Decision / Clinical Rule Application

The model applies the selected clinical rule to the accepted measurement or observation.

For deterministic tasks, verify this programmatically.

Example:

- CTR: 0.55
- threshold: greater than 0.50
- expected interpretation: cardiomegaly supported

The gate verifies that the final decision follows from the accepted evidence.

---

# Cross-Stage Dependency Audit

This is our addition.

Do **not** call it Stage 5.

It is a controller-level audit after the native Path-1 stages.

The audit verifies that the accepted reasoning states are mutually consistent.

For cardiomegaly, check that:

- the criterion is CTR;
- CTR requires the correct cardiac and thoracic anatomy;
- the selected anatomy supports the measurements;
- the measurements support the reported ratio;
- the ratio and threshold support the final diagnosis.

Prefer deterministic dependency rules whenever possible.

Avoid generic LLM judging for relationships that can be checked symbolically.

---

# Controller

Use a simple two-state controller.

Possible outputs:

- `PASS`
- `REPAIR`

Do not implement:

- ACCEPT / REPAIR / REJECT
- abstention
- automatic routing to Path 2

If the retry budget is exhausted, the Path-1 case fails.

---

# Retry Policy

Initial default:

**3 total attempts per stage**

This means:

- 1 initial attempt
- maximum 2 repair attempts

Make this configurable.

Example config:

```yaml
controller:
  max_attempts_per_stage: 3
```

Do not hard-code the number throughout the codebase.

---

# Repair Prompting

Repair prompts must not reveal the answer.

Good:

> The current measurement could not be verified against the accepted anatomical evidence. Re-evaluate this stage and return a revised answer.

Bad:

> Your answer is wrong. The correct CTR is 0.55.

Repair prompts should contain:

- stage identifier
- generic reason for verification failure
- relevant accepted upstream context

They should not contain:

- benchmark answer
- oracle answer
- exact corrected measurement
- downstream diagnosis unless required

---

# Context-Isolated Verification

Each verifier should receive only the information required for that stage.

Possible verifier context:

- original image if needed
- current stage output
- required accepted upstream states

Avoid giving the verifier:

- later-stage outputs
- unnecessary reasoning history
- initial final diagnosis where unnecessary

Create an ablation where the verifier instead receives the complete reasoning history.

---

# Suggested Repository Structure

```text
cxreason-gated/
│
├── README.md
├── requirements.txt
├── configs/
│   ├── default.yaml
│   ├── qwen.yaml
│   └── medgemma.yaml
│
├── cxreason/
│   ├── controller/
│   │   ├── controller.py
│   │   ├── repair.py
│   │   ├── state.py
│   │   └── dependency_audit.py
│   │
│   ├── gates/
│   │   ├── base.py
│   │   ├── criterion.py
│   │   ├── anatomy.py
│   │   ├── measurement.py
│   │   ├── clinical_rule.py
│   │   └── oracle.py
│   │
│   ├── baselines/
│   │   ├── greedy.py
│   │   ├── self_consistency.py
│   │   └── full_restart.py
│   │
│   ├── evaluation/
│   │   ├── path1_runner.py
│   │   ├── metrics.py
│   │   ├── logging.py
│   │   └── statistics.py
│   │
│   ├── corruption/
│   │   ├── anatomy.py
│   │   ├── measurement.py
│   │   ├── clinical_rule.py
│   │   └── dependency.py
│   │
│   └── integrations/
│       └── cxreasonbench.py
│
├── scripts/
│   ├── run_baseline.py
│   ├── run_local_repair.py
│   ├── run_oracle.py
│   ├── run_self_consistency.py
│   ├── run_corruption_eval.py
│   └── run_pilot.py
│
└── tests/
```

---

# Core Interfaces

## Gate Result

```python
from dataclasses import dataclass
from typing import Optional

@dataclass
class GateResult:
    passed: bool
    score: Optional[float] = None
    reason: Optional[str] = None
    metadata: Optional[dict] = None
```

---

## Base Gate

```python
from abc import ABC, abstractmethod

class StageGate(ABC):

    @abstractmethod
    def verify(
        self,
        stage_output,
        context,
    ) -> GateResult:
        pass
```

---

## Controller State

Track:

```python
{
    "case_id": ...,
    "task": ...,
    "initial_answer": ...,
    "stage_outputs": {},
    "accepted_outputs": {},
    "attempts": {},
    "gate_results": {},
    "model_calls": 0,
    "verifier_calls": 0,
    "generated_tokens": 0,
}
```

---

# Local Repair Algorithm

Pseudo-code:

```python
for stage in path1_stages:

    attempts = 0

    while attempts < max_attempts:

        attempts += 1

        output = generate_stage(
            model=model,
            image=image,
            stage=stage,
            accepted_context=accepted_states,
        )

        result = gate.verify(
            stage_output=output,
            context=accepted_states,
        )

        log_attempt(
            stage=stage,
            attempt=attempts,
            output=output,
            gate_result=result,
        )

        if result.passed:
            accepted_states[stage] = output
            break

        if attempts < max_attempts:
            repair_feedback = create_repair_feedback(
                stage=stage,
                gate_result=result,
            )

    if stage not in accepted_states:
        mark_path1_failure()
        break
```

---

# Oracle Verifier

The oracle verifier should use benchmark correctness only to return:

- PASS
- REPAIR

It must never expose:

- correct option
- correct measurement
- correct anatomy
- correct diagnosis

Pseudo-code:

```python
class OracleGate(StageGate):

    def __init__(self, benchmark_scorer):
        self.benchmark_scorer = benchmark_scorer

    def verify(self, stage_output, context):

        correct = self.benchmark_scorer.is_correct(
            stage=context.stage,
            output=stage_output,
            case=context.case,
        )

        return GateResult(
            passed=correct,
            reason=None if correct else "stage_failed",
        )
```

---

# Baselines

Implement the following systems.

## 1. Greedy Baseline

Native CXReasonBench Path 1.

One attempt.

No verification.

No repair.

---

## 2. Structured Prompting Baseline

Use the same stage format as our method.

No external verification.

One generation per stage.

---

## 3. Call-Matched Self-Consistency

Give the model approximately the same generation budget as the repair system.

For discrete answers:

- generate multiple candidates;
- use majority vote.

This controls for extra sampling compute.

---

## 4. Full-Chain Restart

When a verifier rejects one stage:

- discard the current reasoning trajectory;
- restart Path 1 from the beginning.

Match the total generation budget with local repair.

---

## 5. Local Repair

Our proposed method.

Only regenerate the failed stage.

Preserve previously accepted states.

---

## 6. Oracle Local Repair

Same controller as Local Repair.

Replace practical gates with oracle PASS/REPAIR signals.

This measures the upper bound of the repair mechanism.

---

# Verifier Validation

Before trusting the gates, create controlled corruptions.

Use correct structured reasoning examples and deliberately introduce mistakes.

## Corruption Type 1 — Anatomical

Examples:

- wrong structure
- swapped structure
- incorrect laterality
- shifted anatomical reference

## Corruption Type 2 — Measurement

Examples:

- correct component measurements + incorrect ratio
- incorrect derived index
- inconsistent measurement range

## Corruption Type 3 — Clinical Rule

Examples:

- correct measurement + incorrect threshold interpretation
- flipped diagnosis

## Corruption Type 4 — Cross-Stage

Examples:

- correct anatomy but incompatible measurement
- correct measurement but unsupported final decision
- criterion requiring anatomy different from selected anatomy

---

# Verifier Metrics

For each gate report:

- precision
- recall
- F1
- AUROC when continuous scores exist
- false-repair rate

False-repair rate means:

> How often does the gate reject a stage that was actually correct?

This is important because repair can damage already-correct reasoning.

---

# Main Evaluation Metrics

## Primary

### Path-1 Completion

Percentage of cases that successfully complete every required Path-1 stage.

### Reasoning Depth

Average number of correct stages reached before failure.

---

## Mechanism Metrics

### First-Attempt Stage Accuracy

Accuracy before the controller intervenes.

### Post-Repair Stage Accuracy

Accuracy after the allowed repair attempts.

### Repair Success Rate

Percentage of initially incorrect states that become correct after repair.

### Damage Rate

Percentage of initially correct states that become incorrect after intervention.

### Measurement Consistency

Whether later numerical reasoning remains consistent with accepted earlier measurements.

### Decision Alignment

Whether the final answer is consistent with the accepted diagnostic evidence.

---

## Efficiency Metrics

Track:

- VLM calls
- verifier calls
- generated tokens
- regenerated stages
- tokens per case
- tokens per successfully completed case

---

## Oracle-Verifier Gap

Measure the performance difference between:

- oracle local repair
- practical local repair

This separates:

- limitations of the repair mechanism
- limitations of the verifier

---

# Pilot Experiment

Before final experiments, run approximately 50 development cases.

Use the same cases for:

- Qwen2.5-VL-7B
- MedGemma-4B

Measure:

- first-attempt accuracy by stage
- verifier score distributions
- repair success
- damage rate
- model calls
- generated tokens

Use this to determine whether the gates can reliably separate correct from incorrect states.

Do not tune thresholds on the final 1,200 benchmark cases.

---

# Development Protocol

Use a separate development split.

Tune only:

- verifier thresholds
- repair prompt wording
- retry count
- decoding settings
- normalization parameters

Then freeze everything.

Final evaluation should not involve iterative threshold tweaking on the CXReasonBench test set.

Prefer patient-disjoint development data where possible.

---

# Main Experiments

Run both primary models under:

1. Greedy Path 1
2. Structured prompting
3. Call-matched self-consistency
4. Full-chain restart
5. Local repair
6. Oracle local repair

Report:

- Completion
- Reasoning Depth
- stage accuracy
- repair success
- damage rate
- verifier metrics
- compute metrics

---

# Ablations

## No Dependency Audit

Remove the Cross-Stage Dependency Audit.

Question:

> Do cross-stage checks detect additional errors?

---

## Global Restart

Replace local repair with complete trajectory regeneration.

Question:

> Is preserving accepted upstream states useful?

---

## Full-Context Verification

Give the verifier the full reasoning history.

Compare with context-isolated verification.

Question:

> Does unnecessary downstream context affect verifier reliability?

---

## Independent vs Benchmark-Linked Visual Verification

Where possible, compare:

- benchmark-linked verifier
- independent verifier

Question:

> Are improvements merely caused by matching the same system used to construct the benchmark?

---

# Main Hypotheses

## H1

Stage-specific external verification and local repair improve Path-1 Completion and Reasoning Depth beyond greedy inference and call-matched self-consistency.

## H2

Cross-stage dependency checking detects inconsistencies that isolated stage verification misses.

## H3

Local repair achieves similar or better performance than full-chain restart while regenerating less computation.

## H4

On tasks with an independent visual verifier, improvements persist when benchmark-linked visual verification is replaced by independently generated anatomical evidence.

---

# Important Experimental Rules

1. Never reveal the correct answer inside repair feedback.
2. Never silently route failures to Path 2.
3. Keep the VLM frozen.
4. Track every model call.
5. Track every repair attempt.
6. Store first-attempt outputs separately from repaired outputs.
7. Evaluate the gates independently before trusting them.
8. Do not call CXAS an independent verifier.
9. Keep native CXReasonBench Stage-2 outputs for the primary experiment.
10. Match compute when comparing against self-consistency and full restart.
11. Freeze thresholds and prompts before final benchmark evaluation.
12. Keep oracle verification separate from practical verification.

---

# First Implementation Milestone

Implement everything that does **not require restricted CXReasonBench data**:

- controller state
- PASS / REPAIR logic
- retry mechanism
- stage gate interface
- deterministic criterion gate
- arithmetic measurement gate
- clinical-rule gate
- dependency audit
- oracle gate interface
- experiment logging
- token/call accounting
- self-consistency baseline
- full-restart baseline
- mock Path-1 cases
- unit tests

The code should allow real CXReasonBench examples to be plugged in later through a dedicated integration layer.

---

# Second Milestone — Dataset Integration

Once CXReasonBench access is available:

1. Connect the official Path-1 cases.
2. Reproduce published baseline behavior.
3. Implement the oracle scorer.
4. Run the pilot.
5. Calibrate practical gates.
6. Freeze the controller.
7. Run final experiments.

---

# Public Codebase to Use as Reference

Official CXReasonBench repository:

`https://github.com/ttumyche/CXReasonBench`

Use the public evaluation pipeline as the reference for:

- model interfaces
- Path-1 prompts
- Path-1 stage execution
- scoring
- benchmark metrics

Prefer building the new controller in a **separate repository** rather than rewriting the benchmark code directly.

Create a thin integration layer so that upstream CXReasonBench behavior remains easy to reproduce.

---

# Final Research Claim

The intended claim is:

> **Frozen medical VLMs can execute structured chest-X-ray reasoning more reliably when intermediate CXReasonBench Path-1 states are externally checked and failed states are repaired locally, rather than relying on unchecked sequential generation, intrinsic self-correction, generic resampling, or full-chain regeneration.**

The important distinction from prior work is that:

- CXReasonBench provides the stages, but does not use them as active correction points;
- CXReasonAgent uses external tools to derive evidence before language-model reasoning;
- self-correction relies on model-generated feedback;
- our system keeps the VLM responsible for reasoning while external systems act only as stage-specific critics.

The central engineering task is therefore:

> **Build a modular verification-and-repair layer around the existing CXReasonBench Path-1 evaluation pipeline without changing the VLM weights or leaking benchmark answers during repair.**