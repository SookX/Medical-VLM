# MedGemma 4B Setup

The initial target model is configured in:

```text
configs/medgemma.yaml
```

The default model ID is:

```text
google/medgemma-4b-it
```

Google also documents MedGemma 1.5 4B as the newer 4B multimodal line. If you
want that exact variant, change `model.model_id` in `configs/medgemma.yaml`
after accepting the corresponding Hugging Face terms.

## Local Setup

1. Create and activate an environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Accept the MedGemma terms on Hugging Face for the chosen model.
4. Authenticate Hugging Face:

```bash
huggingface-cli login
```

or set `HF_TOKEN` in your shell or `.env.local`.

5. Run the smoke test with a real CXR image:

```bash
python scripts/smoke_medgemma.py --image path/to/chest_xray.png
```

The current repository does not contain actual X-ray images. CheXStruct provides
CSV measurements and labels only; the X-ray pixels must come from NIH, OpenI,
VinDr-CXR, MIMIC-CXR-JPG, or another authorized source.

## Build Order

Start with the training-free framework pieces that do not need official
CXReasonBench cases:

1. Data abstractions: `Path1Case`, `StageOutput`, `AcceptedState`.
2. Gates: criterion, arithmetic measurement, clinical rule.
3. Controller: stage retry loop, local repair prompts, accounting.
4. Mock cases: cardiomegaly first, because CheXStruct exposes CTR components.
5. MedGemma adapter: one method that receives an image, stage ID, and accepted
   upstream context, then returns raw text.
6. Upstream CXReasonBench adapter: plug in official Path-1 prompts and scoring
   once the restricted dataset is available.

