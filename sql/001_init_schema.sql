begin;

create extension if not exists pgcrypto;
create extension if not exists vector with schema extensions;

create schema if not exists skg;

create table if not exists skg.nodes (
    id text primary key,
    type text not null,
    label text not null,
    status text not null default 'asserted',
    domain text,
    description text,
    properties jsonb not null default '{}'::jsonb,
    metadata jsonb not null default '{}'::jsonb,
    provenance jsonb not null default '[]'::jsonb,
    embedding extensions.vector,
    embedding_model text,
    embedding_dimensions integer,
    fts_document tsvector generated always as (
        setweight(to_tsvector('simple', coalesce(label, '')), 'A') ||
        setweight(to_tsvector('simple', coalesce(description, '')), 'B') ||
        setweight(to_tsvector('simple', coalesce(properties->>'text', '')), 'C')
    ) stored,
    created_at timestamptz not null default now(),
    updated_at timestamptz not null default now(),
    constraint nodes_id_ck check (id ~ '^n_[a-f0-9]{64}$'),
    constraint nodes_status_ck check (status in ('asserted','candidate','inferred','deprecated','rejected')),
    constraint nodes_properties_object_ck check (jsonb_typeof(properties) = 'object'),
    constraint nodes_metadata_object_ck check (jsonb_typeof(metadata) = 'object'),
    constraint nodes_provenance_array_ck check (jsonb_typeof(provenance) = 'array'),
    constraint nodes_embedding_contract_ck check (
        embedding is null
        or (embedding_model is not null and embedding_dimensions is not null and embedding_dimensions > 0)
    )
);

create table if not exists skg.evidence (
    id text primary key,
    kind text not null,
    source_class text not null,
    source text not null,
    locator text,
    content_hash text not null,
    hash_algorithm text not null default 'sha256',
    verified boolean not null default false,
    observed_at timestamptz,
    metadata jsonb not null default '{}'::jsonb,
    created_at timestamptz not null default now(),
    constraint evidence_id_ck check (id ~ '^ev_[a-f0-9]{64}$'),
    constraint evidence_hash_ck check (content_hash ~ '^[a-f0-9]{64}$'),
    constraint evidence_hash_algorithm_ck check (hash_algorithm = 'sha256'),
    constraint evidence_metadata_object_ck check (jsonb_typeof(metadata) = 'object')
);

create table if not exists skg.edges (
    id text primary key,
    source text not null references skg.nodes(id) on delete restrict,
    target text not null references skg.nodes(id) on delete restrict,
    type text not null,
    status text not null default 'candidate',
    properties jsonb not null default '{}'::jsonb,
    metadata jsonb not null default '{}'::jsonb,
    score jsonb not null default '{}'::jsonb,
    reason jsonb not null default '{}'::jsonb,
    provenance jsonb not null default '[]'::jsonb,
    confidence double precision,
    origin text not null default 'asserted',
    created_at timestamptz not null default now(),
    updated_at timestamptz not null default now(),
    constraint edges_id_ck check (id ~ '^e_[a-f0-9]{64}$'),
    constraint edges_status_ck check (status in ('candidate','validated','rejected','deprecated')),
    constraint edges_origin_ck check (origin in ('asserted','inferred','model_assisted','imported')),
    constraint edges_confidence_ck check (confidence is null or confidence between 0 and 1),
    constraint edges_properties_object_ck check (jsonb_typeof(properties) = 'object'),
    constraint edges_metadata_object_ck check (jsonb_typeof(metadata) = 'object'),
    constraint edges_score_object_ck check (jsonb_typeof(score) = 'object'),
    constraint edges_reason_object_ck check (jsonb_typeof(reason) = 'object'),
    constraint edges_provenance_array_ck check (jsonb_typeof(provenance) = 'array'),
    constraint edges_no_self_ref_ck check (source <> target)
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

create table if not exists skg.audit_events (
    id bigint generated always as identity primary key,
    event_type text not null,
    object_type text not null,
    object_id text not null,
    details jsonb not null default '{}'::jsonb,
    actor_role text not null default current_user,
    created_at timestamptz not null default now()
);

create index if not exists idx_nodes_type on skg.nodes(type);
create index if not exists idx_nodes_status on skg.nodes(status);
create index if not exists idx_nodes_domain on skg.nodes(domain);
create index if not exists idx_edges_source on skg.edges(source);
create index if not exists idx_edges_target on skg.edges(target);
create index if not exists idx_edges_type on skg.edges(type);
create index if not exists idx_edges_status on skg.edges(status);
create index if not exists idx_nodes_properties_gin on skg.nodes using gin(properties);
create index if not exists idx_nodes_metadata_gin on skg.nodes using gin(metadata);
create index if not exists idx_edges_properties_gin on skg.edges using gin(properties);
create index if not exists idx_edges_metadata_gin on skg.edges using gin(metadata);
create index if not exists idx_edges_score_gin on skg.edges using gin(score);
create index if not exists idx_nodes_fts_gin on skg.nodes using gin(fts_document);
create index if not exists idx_node_evidence_node on skg.node_evidence(node_id);
create index if not exists idx_node_evidence_evidence on skg.node_evidence(evidence_id);
create index if not exists idx_edge_evidence_edge on skg.edge_evidence(edge_id);
create index if not exists idx_edge_evidence_evidence on skg.edge_evidence(evidence_id);
create index if not exists idx_audit_events_object on skg.audit_events(object_type, object_id);
create index if not exists idx_audit_events_created on skg.audit_events(created_at desc);

-- Vector dimensions are intentionally unconstrained here. They are enforced per embedding model contract at ingestion time.

create table if not exists skg.embedding_spaces (
    model_name text primary key,
    dimensions integer not null check (dimensions > 0),
    distance text not null default 'cosine' check (distance in ('cosine','l2','inner_product')),
    version text not null,
    active boolean not null default true,
    created_at timestamptz not null default now()
);

create index if not exists idx_nodes_embedding_model on skg.nodes(embedding_model, embedding_dimensions);

create or replace function skg.touch_updated_at()
returns trigger language plpgsql as $$
begin
  new.updated_at = now();
  return new;
end;
$$;

drop trigger if exists trg_nodes_touch on skg.nodes;
create trigger trg_nodes_touch before update on skg.nodes for each row execute function skg.touch_updated_at();
drop trigger if exists trg_edges_touch on skg.edges;
create trigger trg_edges_touch before update on skg.edges for each row execute function skg.touch_updated_at();

-- Default-deny posture for API-facing roles.
alter table skg.nodes enable row level security;
alter table skg.edges enable row level security;
alter table skg.evidence enable row level security;
alter table skg.audit_events enable row level security;

revoke all on schema skg from public;
revoke all on all tables in schema skg from public;

-- Service role access is explicit; owner/postgres remains available for migrations.
grant usage on schema skg to service_role;
grant select, insert, update, delete on skg.nodes, skg.edges, skg.evidence to service_role;
grant select on skg.audit_events to service_role;

drop policy if exists service_nodes on skg.nodes;
create policy service_nodes on skg.nodes for all to service_role using (true) with check (true);
drop policy if exists service_edges on skg.edges;
create policy service_edges on skg.edges for all to service_role using (true) with check (true);
drop policy if exists service_evidence on skg.evidence;
create policy service_evidence on skg.evidence for all to service_role using (true) with check (true);
drop policy if exists service_audit on skg.audit_events;
create policy service_audit on skg.audit_events for select to service_role using (true);

commit;
