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
  "verification_steps": ["How a person can independently check the answer."],
  "lvu_agreement": 0.95,
  "lvu_label": "FACT",
  "peer_prediction_consistent": true,
  "peer_prediction_labels": ["FACT", "FACT", "FACT"]
}
```

### Field definitions

- `lvu_agreement`: Agreement score across 5 samples (0.0–1.0). ≥0.9 = FACT, ≥0.6 = INFERENCE, <0.6 = UNKNOWN.
- `lvu_label`: Epistemic label derived from LVU consistency (may differ from self-reported label).
- `peer_prediction_consistent`: true if all 3 paraphrased prompts yield the same epistemic label.
- `peer_prediction_labels`: List of epistemic labels from each paraphrased prompt.

## Validation rules

1. The envelope must be validated as JSON. Invalid or unsupported output is `UNKNOWN`, not silently repaired into a confident answer.
2. If `label == 'FACT'`, then `source_ids` must be non-empty and all IDs must exist in the provenance manifest.
3. If `lvu_agreement < 0.6`, then `label` must be `UNKNOWN` (override self-reported label).
4. If `peer_prediction_consistent == false`, then `label` must be `CONFLICTED`.
5. If `source_ids` is empty and the query is health/legal/safety-critical, `label` must be `UNKNOWN`.

## Example outputs

### FACT (high confidence, verified)

```json
{
  "label": "FACT",
  "answer": "Water boils at 100 °C at standard atmospheric pressure (1 atm).",
  "source_ids": ["sha256:abc123..."],
  "limitations": [],
  "verification_steps": ["Check any physics or chemistry textbook, or NIST reference data."],
  "lvu_agreement": 1.0,
  "lvu_label": "FACT",
  "peer_prediction_consistent": true,
  "peer_prediction_labels": ["FACT", "FACT", "FACT"]
}
```

### UNKNOWN (honest abstention)

```json
{
  "label": "UNKNOWN",
  "answer": "There is insufficient peer-reviewed evidence to determine the long-term efficacy of this treatment.",
  "source_ids": [],
  "limitations": ["No randomized controlled trials with >5 year follow-up found in PubMed or Cochrane."],
  "verification_steps": ["Search PubMed for systematic reviews on this treatment; check Cochrane Library."],
  "lvu_agreement": 0.4,
  "lvu_label": "UNKNOWN",
  "peer_prediction_consistent": true,
  "peer_prediction_labels": ["UNKNOWN", "UNKNOWN", "UNKNOWN"]
}
```

### CONFLICTED (sources disagree)

```json
{
  "label": "CONFLICTED",
  "answer": "Some studies suggest benefit, while others show no effect or potential harm.",
  "source_ids": ["sha256:abc123...", "sha256:def456..."],
  "limitations": ["Meta-analyses disagree; sample sizes vary widely."],
  "verification_steps": ["Compare Cochrane review (2024) with individual RCTs in PubMed."],
  "lvu_agreement": 0.5,
  "lvu_label": "UNKNOWN",
  "peer_prediction_consistent": false,
  "peer_prediction_labels": ["INFERENCE", "UNKNOWN", "CONFLICTED"]
}
```
