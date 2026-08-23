# Epistemic output schema

A label is a statement about support, not a guarantee of truth. The model must be evaluated against held-out sources and adversarial prompts before any safety-critical use.

## Labels

- `FACT`: Directly supported by one or more cited, accessible sources; cite the source IDs.
- `INFERENCE`: A conclusion derived from stated facts; explain the reasoning and alternatives.
- `UNKNOWN`: The available evidence is insufficient, inaccessible, or not verified.
- `CONFLICTED`: Credible sources disagree; identify the disagreement and avoid choosing silently.

## Required answer envelope

```json
{
  "label": "FACT | INFERENCE | UNKNOWN | CONFLICTED",
  "answer": "Plain answer without emotional simulation or marketing.",
  "source_ids": ["sha256-or-stable-id"],
  "limitations": ["What is missing, uncertain, or not verified."],
  "verification_steps": ["How a person can independently check the answer."]
}
```

The envelope must be validated as JSON. Invalid or unsupported output is `UNKNOWN`, not silently repaired into a confident answer.
