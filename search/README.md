# Niyah Search Core

Local-first web retrieval primitives. External search providers are optional adapters.

Implemented pieces are deliberately narrow: crawl frontier, URL normalization, robots policy boundary, document storage contracts, inverted index, BM25 scoring, and source/evidence metadata.

The crawler must obey robots.txt and explicit fetch limits. The search index is not a claim of web-scale coverage until populated and measured.
