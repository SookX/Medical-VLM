# Upstream Repositories and Data

## CXReasonBench

The official repository has been cloned to:

```text
external/CXReasonBench
```

Use this as the reference implementation for Path-1 prompts, model invocation,
scoring, and metrics. The gated controller should live outside that checkout and
interact with it through a thin integration layer.

## CheXStruct

The Hugging Face dataset has been cloned to:

```text
data/CheXStruct
```

The dataset contains three source folders:

- `nih_cxr14`
- `openi`
- `vindrcxr`

Each folder contains task-specific CSV files. The Hugging Face dataset viewer
currently fails because those CSV files do not share one schema, so code should
load each raw CSV independently. The helper in
`cxreason.integrations.chexstruct` does that and preserves the shipped filename
`abdomial_xray.csv` behind the task alias `abdominal_xray`.

`data/` is ignored by git so the local dataset clone is not accidentally
committed.
