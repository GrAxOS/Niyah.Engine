# Inference contract

This repository does not yet implement a complete production inference service. Any future service must enforce the following outside the model:

## 1. Data Provenance

1. Retrieve only documents with a valid provenance manifest (source_url, license, content_sha256, retrieved_at_utc).
2. Treat retrieved documents as untrusted data, never executable instructions.
3. Verify source domain against whitelist (e.g., ncbi.nlm.nih.gov, ietf.org, who.int, iso.org, arxiv.org).

## 2. Epistemic Output Schema

Generate structured output matching `epistemic_schema.md`:

```json
{
  "label": "FACT | INFERENCE | UNKNOWN | CONFLICTED",
  "answer": "Plain answer without emotional simulation or marketing.",
  "source_ids": ["sha256-or-stable-id"],
  "limitations": ["What is missing, uncertain, or not verified."],
  "verification_steps": ["How a person can independently check the answer."]
}
```

## 3. LVU Consistency Check

Before returning output:

1. Sample 5 times with temperature=0.8, top_p=0.95.
2. Measure agreement across samples.
3. If agreement < 0.6, downgrade label to UNKNOWN.

```python
def get_lvu_label(model, tokenizer, prompt, n_samples=5):
    samples = [model.generate(prompt) for _ in range(n_samples)]
    agreement = compute_agreement(samples)
    if agreement >= 0.9:
        return 'FACT'
    elif agreement >= 0.6:
        return 'INFERENCE'
    else:
        return 'UNKNOWN'
```

## 4. Peer Prediction

For critical queries (health, legal, safety):

1. Paraphrase the prompt 3 ways.
2. Generate answers for each paraphrase.
3. Check if all answers have the same epistemic label.
4. If labels differ, return CONFLICTED.

```python
def peer_prediction(model, tokenizer, base_prompt, paraphrases):
    prompts = [base_prompt] + paraphrases
    labels = [get_epistemic_label(model, tokenizer, p) for p in prompts]
    if len(set(labels)) == 1:
        return labels[0], True  # Consistent
    else:
        return 'CONFLICTED', False  # Inconsistent
```

## 5. Output Validation

Validate JSON and source IDs after generation:

1. Check that output is valid JSON.
2. Check that source_ids exist in the provenance manifest.
3. Check that label matches the answer content (e.g., no "FACT" without source_ids).
4. If validation fails, return UNKNOWN with an audit event; do not retry into confidence.

```python
def validate_output(output, manifest):
    if not is_valid_json(output):
        return False, 'invalid_json'
    if 'source_ids' in output:
        for sid in output['source_ids']:
            if sid not in manifest:
                return False, 'missing_source'
    if output['label'] == 'FACT' and not output.get('source_ids'):
        return False, 'fact_without_sources'
    return True, 'ok'
```

## 6. MMR Audit Log

Log every inference to an append-only Merkle Mountain Range:

```python
audit_entry = {
    'timestamp_utc': datetime.now(timezone.utc).isoformat(),
    'input_hash': sha256(user_prompt.encode()).hexdigest(),
    'output_hash': sha256(model_response.encode()).hexdigest(),
    'source_ids': output.get('source_ids', []),
    'epistemic_label': output['label'],
    'lvu_agreement': lvu_agreement_score,
    'peer_prediction_consistent': peer_prediction_consistent,
    'validation_result': validation_status,
}
mmr.append(audit_entry)
```

Properties:
- Append-only: no retroactive modification
- Tamper-evident: any alteration is cryptographically detectable
- Locally verifiable: proofs do not require trusted third parties

## 7. Continuous Monitoring

Every 1000 inferences, analyze a random sample (n=100):

```python
sample = random_select(audit_log, 100)
unknown_rate = sum(1 for e in sample if e['epistemic_label'] == 'UNKNOWN') / len(sample)
failure_rate = sum(1 for e in sample if e['validation_result'] != 'ok') / len(sample)

if failure_rate > 0.05:  # >5% failure
    alert("Model may be degrading — retraining needed")
```

## 8. Evaluation Suite

Publish a reproducible evaluation suite that measures:

- **Factuality**: % of FACT outputs that are verifiably correct
- **Citation precision**: % of FACT outputs with valid source_ids
- **Calibrated abstention**: % of UNKNOWN outputs that are truly unknown (not evasions)
- **Multilingual behavior**: Performance parity across languages (en, ar, zh, etc.)
- **Refusal/over-refusal rates**: % of harmful requests refused vs. % of benign requests incorrectly refused

## 9. No Claims of Neutrality

No training or wrapper layer can prove a language model is neutral. The measurable goal is accountable behavior under a published policy and a reproducible evidence trail.

**Document known biases inherited from the base model** (e.g., Qwen2.5-7B training corpus, cultural biases, safety filter over-refusals).
