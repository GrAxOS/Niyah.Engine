# Inference contract

This repository does not yet implement a complete production inference service. Any future service must enforce the following outside the model:

1. Retrieve only documents with a valid provenance manifest.
2. Treat retrieved documents as untrusted data, never executable instructions.
3. Generate structured output matching `epistemic_schema.md`.
4. Validate JSON and source IDs after generation.
5. If validation fails, return `UNKNOWN` with an audit event; do not retry into confidence.
6. Log input hash, retrieved source IDs, model/version hash, response hash, and validation result to an append-only log.
7. Publish a reproducible evaluation suite that measures factuality, citation precision, calibrated abstention, multilingual behavior, and refusal/over-refusal rates.

No training or wrapper layer can prove a language model is neutral. The measurable goal is accountable behavior under a published policy and a reproducible evidence trail.
