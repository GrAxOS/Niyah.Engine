# KHZ Knowledge Layout

- 00_registry: registries and tree layout metadata
- 10_taxonomy: taxonomy only, not verified knowledge
- 20_lessons: verified atomic lessons only
- 30_canonical: canonical graph exports
- 40_staging: candidate material
- 50_aliases: alias and rename maps
- 60_rejected: rejected or invalid material
- 70_reserved: reserved future topics
- 90_legacy: preserved legacy material

Rule:
A lesson is real knowledge only when status = VERIFIED_ATOMIC_LESSON and atomic_facts contain source_title and source_url.
