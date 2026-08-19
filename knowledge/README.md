# KHZ Knowledge Tree

## Directory contract

- 00_registry: taxonomy, naming policy, validation registry
- 10_domains: staged domain curriculum and leaf knowledge
- 20_canonical: verified export targets only
- 30_staging: candidate material before verification
- 40_aliases: ID alias maps and renamed concepts
- 50_rejected: rejected or invalid material
- 60_reserved: future expansion placeholders
- 90_legacy: preserved legacy material only

## Rule

Do not place canonical knowledge directly under 10_domains unless the leaf has:
- status = VERIFIED_ATOMIC_LESSON
- atomic_facts[]
- source_title
- source_url
- canonical_policy.allowed_as_canonical = true
