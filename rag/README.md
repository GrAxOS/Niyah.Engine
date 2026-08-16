# Native RAG source layer

This directory contains source policy and seed metadata for local retrieval.

## Source policy

1. Prefer primary sources: standards bodies, official project documentation, original research repositories, official source trees, and first-party vendor documentation.
2. Keep secondary references available when useful, but record them as secondary.
3. Store the canonical URI and a SHA-256 content hash after retrieval in the local evidence database.
4. Do not treat a search snippet, model response, cached answer, or provider label as proof.
5. Preserve conflicting sources instead of silently selecting one.
6. Respect robots.txt, HTTP limits, redirects, TLS verification, and local fetch policy.
7. Retrieval is local-first. External search providers are adapters, not the RAG data plane.
8. The source registry contains real source URLs only; it is not a claim that those sites have already been crawled into the local index.

The initial registry in `official_sources.json` uses current first-party technical sources such as IETF/RFC Editor, NIST, Linux kernel documentation, LLVM, GCC, Microsoft Learn, Python documentation, Mozilla MDN, and cppreference.
