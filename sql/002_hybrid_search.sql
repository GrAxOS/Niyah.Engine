begin;

create or replace function skg.hybrid_search(
    query_text text,
    query_embedding extensions.vector,
    top_k integer default 20,
    semantic_candidates integer default 50,
    lexical_candidates integer default 50,
    rrf_k integer default 50,
    embedding_model_name text default null,
    filter_domain text default null
)
returns table (
    id text,
    type text,
    label text,
    status text,
    domain text,
    lexical_rank bigint,
    semantic_rank bigint,
    rrf_score double precision
)
language sql
stable
as $$
with lexical as (
    select
        n.id,
        row_number() over (
            order by ts_rank_cd(
                n.fts_document,
                websearch_to_tsquery('simple', query_text)
            ) desc,
            n.id
        ) as rank_ix
    from skg.nodes n
    where query_text is not null
      and btrim(query_text) <> ''
      and n.status not in ('rejected','deprecated')
      and n.fts_document @@ websearch_to_tsquery('simple', query_text)
      and (filter_domain is null or n.domain = filter_domain)
      and (
          embedding_model_name is null
          or n.embedding_model is null
          or n.embedding_model = embedding_model_name
      )
    order by rank_ix
    limit greatest(1, least(lexical_candidates, 1000))
),
semantic as (
    select
        n.id,
        row_number() over (
            order by n.embedding <=> query_embedding,
                     n.id
        ) as rank_ix
    from skg.nodes n
    where query_embedding is not null
      and n.embedding is not null
      and n.status not in ('rejected','deprecated')
      and (filter_domain is null or n.domain = filter_domain)
      and (embedding_model_name is null or n.embedding_model = embedding_model_name)
    order by n.embedding <=> query_embedding, n.id
    limit greatest(1, least(semantic_candidates, 1000))
),
fused as (
    select
        coalesce(l.id, s.id) as id,
        l.rank_ix as lexical_rank,
        s.rank_ix as semantic_rank,
        coalesce(1.0 / (rrf_k + l.rank_ix), 0.0) +
        coalesce(1.0 / (rrf_k + s.rank_ix), 0.0) as rrf_score
    from lexical l
    full outer join semantic s on s.id = l.id
)
select
    n.id,
    n.type,
    n.label,
    n.status,
    n.domain,
    f.lexical_rank,
    f.semantic_rank,
    f.rrf_score
from fused f
join skg.nodes n on n.id = f.id
order by f.rrf_score desc, n.id
limit greatest(1, least(top_k, 200));
$$;

comment on function skg.hybrid_search is
'Local hybrid lexical + pgvector retrieval fused by Reciprocal Rank Fusion. Query/filter parameters are applied before ranking.';

commit;
