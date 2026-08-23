# Data source policy

Niyah.Neutral must keep source identity, licence, retrieval date, checksum, language, domain, and transformation history for every document. A source is not accepted merely because it is scientific or official; it must have a documented licence or permission compatible with the intended use.

## Admission classes

- Primary: standards, official guidance, public-domain government publications, peer-reviewed articles whose full-text licence permits the intended use.
- Secondary: reputable technical documentation and textbooks with documented rights.
- Excluded by default: social feeds, dating/parasocial material, advertising, scraped personal data, unverifiable claims, synthetic text without provenance, and documents with unclear reuse rights.

## Initial candidate registries

| Domain | Registry | Status before ingest |
|---|---|---|
| Medicine | PubMed Central Open Access Subset | Candidate; retain each article licence and provenance |
| Public health | WHO publications | Candidate; review terms per document |
| Drug labels | DailyMed | Candidate; retain label version and date |
| Internet and networking | IETF RFC Editor | Candidate; retain RFC number, status, and errata state |
| Computing | POSIX / The Open Group documentation | Candidate; review reuse terms |
| Research | arXiv | Candidate; accept only records with explicit compatible licence; peer review is not assumed |
| Mathematics and physics | Open-access journals and repositories | Candidate; require licence and provenance per item |

## Required manifest fields

Each admitted document is stored as JSONL with: `document_id`, `source_name`, `source_url`, `retrieved_at_utc`, `license`, `content_sha256`, `language`, `domain`, `publication_date`, `version`, `transformations`, `quality_flags`, and `text`.

## Non-negotiable rules

1. Do not label a source as peer-reviewed unless its review status is independently recorded.
2. Do not remove citations, provenance, safety-critical caveats, or uncertainty language during cleaning.
3. Do not train from a document until licence and checksum are present.
4. Keep rejected records and rejection reasons in a separate immutable manifest.
5. No claim of neutrality: publish the corpus manifest, exclusions, and evaluation results instead.
