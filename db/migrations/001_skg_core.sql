create schema if not exists skg;
create schema if not exists skg_search;
create schema if not exists skg_audit;

create extension if not exists vector;

create table if not exists skg.nodes (
    id text primary key,
    node_type text not null,
    label text not null,
    description text,
    status text not null default 'asserted',
    scope_domain text,
    scope_jurisdiction text,
    scope_environment text,
    properties jsonb not null default '{}'::jsonb,
    tags text[] not null default '{}',
    aliases text[] not null default '{}',
    confidence_overall double precision,
    confidence_semantic double precision,
    confidence_source double precision,
    confidence_structural double precision,
    embedding_model text,
    embedding_dimensions integer,
    embedding_version text,
    embedding vector,
    valid_from timestamptz,
    valid_to timestamptz,
    created_at timestamptz not null default now(),
    updated_at timestamptz not null default now(),
    metadata jsonb not null default '{}'::jsonb,
    constraint node_id_ck check (id ~ '^n_[a-f0-9]{64}$'),
    constraint node_status_ck check (status in ('asserted','candidate','inferred','deprecated','rejected')),
    constraint node_confidence_ck check (
      confidence_overall is null or confidence_overall between 0 and 1
    ),
    constraint node_properties_object_ck check (jsonb_typeof(properties) = 'object'),
    constraint node_metadata_object_ck check (jsonb_typeof(metadata) = 'object'),
    constraint node_time_ck check (valid_to is null or valid_from is null or valid_to >= valid_from)
);

create table if not exists skg.edges (
    id text primary key,
    source_id text not null references skg.nodes(id) on delete restrict,
    target_id text not null references skg.nodes(id) on delete restrict,
    edge_type text not null,
    direction text not null default 'directed',
    status text not null default 'candidate',
    origin text not null default 'inferred',
    properties jsonb not null default '{}'::jsonb,
    score jsonb not null default '{}'::jsonb,
    confidence_overall double precision,
    confidence_semantic double precision,
    confidence_source double precision,
    confidence_structural double precision,
    reason jsonb not null default '{}'::jsonb,
    valid_from timestamptz,
    valid_to timestamptz,
    created_at timestamptz not null default now(),
    updated_at timestamptz not null default now(),
    metadata jsonb not null default '{}'::jsonb,
    constraint edge_id_ck check (id ~ '^e_[a-f0-9]{64}$'),
    constraint edge_status_ck check (status in ('candidate','validated','rejected','deprecated')),
    constraint edge_origin_ck check (origin in ('asserted','inferred','model_assisted','imported')),
    constraint edge_direction_ck check (direction in ('directed','symmetric')),
    constraint edge_self_loop_ck check (source_id <> target_id),
    constraint edge_time_ck check (valid_to is null or valid_from is null or valid_to >= valid_from)
);

create table if not exists skg.evidence (
    id text primary key,
    evidence_type text not null,
    source text not null,
    locator text,
    excerpt_hash text,
    content_hash text not null,
    observed_at timestamptz not null,
    verification jsonb not null default '{}'::jsonb,
    metadata jsonb not null default '{}'::jsonb,
    constraint evidence_id_ck check (id ~ '^ev_[a-f0-9]{64}$'),
    constraint evidence_content_hash_ck check (content_hash ~ '^[a-f0-9]{64}$')
);

create table if not exists skg.node_evidence (
    node_id text not null references skg.nodes(id) on delete cascade,
    evidence_id text not null references skg.evidence(id) on delete restrict,
    role text not null default 'supporting',
    primary key (node_id, evidence_id)
);

create table if not exists skg.edge_evidence (
    edge_id text not null references skg.edges(id) on delete cascade,
    evidence_id text not null references skg.evidence(id) on delete restrict,
    role text not null default 'supporting',
    primary key (edge_id, evidence_id)
);

create table if not exists skg_audit.events (
    event_id bigint generated always as identity primary key,
    event_type text not null,
    object_type text not null,
    object_id text not null,
    before_hash text,
    after_hash text,
    evidence_id text references skg.evidence(id) on delete restrict,
    created_at timestamptz not null default now(),
    metadata jsonb not null default '{}'::jsonb
);

create index if not exists idx_nodes_type_status on skg.nodes(node_type, status);
create index if not exists idx_nodes_domain on skg.nodes(scope_domain);
create index if not exists idx_nodes_properties_gin on skg.nodes using gin(properties);
create index if not exists idx_nodes_tags_gin on skg.nodes using gin(tags);
create index if not exists idx_edges_source_type on skg.edges(source_id, edge_type, status);
create index if not exists idx_edges_target_type on skg.edges(target_id, edge_type, status);
create index if not exists idx_edges_properties_gin on skg.edges using gin(properties);
create index if not exists idx_nodes_fts on skg.nodes using gin(
  to_tsvector('simple', coalesce(label,'') || ' ' || coalesce(description,''))
);

-- Keep vector dimensionality model-specific. Create a concrete vector index only after
-- selecting an embedding contract. Example for 1536 dimensions:
-- create index idx_nodes_embedding_hnsw on skg.nodes using hnsw ((embedding::vector(1536)) vector_cosine_ops);

create or replace function skg.hybrid_node_search(
    p_query text,
    p_embedding vector,
    p_match_count integer default 20,
    p_rrf_k integer default 50
)
returns table (
    node_id text,
    label text,
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
                to_tsvector('simple', coalesce(n.label,'') || ' ' || coalesce(n.description,'')),
                websearch_to_tsquery('simple', p_query)
            ) desc,
            n.id
        ) as rank_ix
    from skg.nodes n
    where n.status in ('asserted','inferred')
      and to_tsvector('simple', coalesce(n.label,'') || ' ' || coalesce(n.description,''))
          @@ websearch_to_tsquery('simple', p_query)
    limit least(p_match_count * 2, 100)
),
semantic as (
    select
        n.id,
        row_number() over (
            order by n.embedding <=> p_embedding,
            n.id
        ) as rank_ix
    from skg.nodes n
    where n.status in ('asserted','inferred')
      and n.embedding is not null
    order by n.embedding <=> p_embedding, n.id
    limit least(p_match_count * 2, 100)
)
select
    coalesce(l.id, s.id) as node_id,
    n.label,
    l.rank_ix,
    s.rank_ix,
    coalesce(1.0 / (p_rrf_k + l.rank_ix), 0) +
    coalesce(1.0 / (p_rrf_k + s.rank_ix), 0) as rrf_score
from lexical l
full outer join semantic s on s.id = l.id
join skg.nodes n on n.id = coalesce(l.id, s.id)
order by rrf_score desc, n.id
limit least(p_match_count, 200);
$$;
