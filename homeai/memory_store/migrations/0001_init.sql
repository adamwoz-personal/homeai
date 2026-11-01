-- 0001_init: long-term memory schema. See plans/MEMORY_PLAN.md §4.
-- Applied by `homeai-memory migrate` in one transaction.

CREATE SCHEMA IF NOT EXISTS memory;

-- Roles are a backstop against application bugs, not a wall: every process
-- runs as the same Unix user. Each process SETs its role after connecting.
DO $$
BEGIN
  IF NOT EXISTS (SELECT FROM pg_roles WHERE rolname = 'homeai_app') THEN
    CREATE ROLE homeai_app NOLOGIN; END IF;
  IF NOT EXISTS (SELECT FROM pg_roles WHERE rolname = 'homeai_mcp') THEN
    CREATE ROLE homeai_mcp NOLOGIN; END IF;
  IF NOT EXISTS (SELECT FROM pg_roles WHERE rolname = 'homeai_ro') THEN
    CREATE ROLE homeai_ro NOLOGIN; END IF;
END $$;

-- Who may see a row. Fails closed: with no person set, nothing is visible.
-- 'household' rows are shared with every identified person.
CREATE FUNCTION memory.can_see(owner text) RETURNS boolean
LANGUAGE sql STABLE AS $$
  SELECT coalesce(current_setting('app.scope', true), '') = 'admin'
      OR (coalesce(current_setting('app.username', true), '') <> ''
          AND owner IN (current_setting('app.username', true), 'household'))
$$;

CREATE TABLE memory.people (
  username       text PRIMARY KEY CHECK (username ~ '^[a-z0-9_]{1,32}$'),
  display_name   text NOT NULL,
  is_child       boolean NOT NULL DEFAULT false,
  memory_enabled boolean NOT NULL DEFAULT true,
  is_admin       boolean NOT NULL DEFAULT false,
  created_at     timestamptz NOT NULL DEFAULT now()
);
INSERT INTO memory.people (username, display_name, memory_enabled) VALUES
  ('household', 'Household', true), ('guest', 'Guest', false);

CREATE TABLE memory.identities (
  channel  text NOT NULL CHECK (channel IN ('voice','sms','email','camera','cli','mcp')),
  handle   text NOT NULL,
  username text NOT NULL REFERENCES memory.people ON DELETE CASCADE ON UPDATE CASCADE,
  verified boolean NOT NULL DEFAULT false,
  PRIMARY KEY (channel, handle)
);

CREATE TABLE memory.episodes (
  id              uuid NOT NULL DEFAULT uuidv7(),
  occurred_at     timestamptz NOT NULL,
  username        text NOT NULL REFERENCES memory.people ON DELETE CASCADE ON UPDATE CASCADE,
  confidence      real NOT NULL CHECK (confidence BETWEEN 0 AND 1),
  channel         text NOT NULL CHECK (channel IN ('voice','sms','email','camera','cli','mcp','system')),
  trust           text NOT NULL CHECK (trust IN ('owner','household','outside','system')),
  session_id      uuid,
  kind            text NOT NULL CHECK (kind IN ('exchange','message','email','event','summary')),
  content         jsonb NOT NULL,
  text_for_search text NOT NULL CHECK (length(text_for_search) <= 4000),
  embedding       vector(384),
  embed_model     text,
  fts             tsvector GENERATED ALWAYS AS (to_tsvector('english', text_for_search)) STORED,
  state           text NOT NULL DEFAULT 'active' CHECK (state IN ('active','summarised','archived')),
  source_ref      text,          -- e.g. transcript ts, for idempotent back-fill
  PRIMARY KEY (id, occurred_at)
) PARTITION BY RANGE (occurred_at);
CREATE INDEX ON memory.episodes (username, occurred_at DESC);
CREATE INDEX ON memory.episodes USING gin (fts);
CREATE INDEX ON memory.episodes USING hnsw (embedding vector_cosine_ops);
CREATE UNIQUE INDEX episodes_source_ref ON memory.episodes (source_ref, occurred_at)
  WHERE source_ref IS NOT NULL;

-- Yearly partitions; a default catches anything outside them (clock errors).
CREATE FUNCTION memory.ensure_partition(year int) RETURNS void
LANGUAGE plpgsql AS $$
BEGIN
  EXECUTE format(
    'CREATE TABLE IF NOT EXISTS memory.episodes_%s PARTITION OF memory.episodes
       FOR VALUES FROM (%L) TO (%L)',
    year, make_date(year, 1, 1), make_date(year + 1, 1, 1));
END $$;
SELECT memory.ensure_partition(y) FROM generate_series(2025, 2030) AS y;
CREATE TABLE memory.episodes_default PARTITION OF memory.episodes DEFAULT;

CREATE TABLE memory.facts (
  id            uuid PRIMARY KEY DEFAULT uuidv7(),
  username      text NOT NULL REFERENCES memory.people ON DELETE CASCADE ON UPDATE CASCADE,
  topics        text[] NOT NULL DEFAULT '{}',
  detail        jsonb NOT NULL DEFAULT '{"v":1}',
  kind          text NOT NULL CHECK (kind IN ('fact','preference','event','question','answer',
                  'thought','opinion','plan','summary','profile','instruction')),
  summary       text NOT NULL CHECK (length(summary) BETWEEN 1 AND 300),
  embedding     vector(384),
  embed_model   text NOT NULL,
  fts           tsvector GENERATED ALWAYS AS (to_tsvector('english', summary)) STORED,
  importance    real NOT NULL DEFAULT 0.5 CHECK (importance BETWEEN 0 AND 1),
  pinned        boolean NOT NULL DEFAULT false,
  trust         text NOT NULL CHECK (trust IN ('owner','household','outside','system')),
  state         text NOT NULL DEFAULT 'active' CHECK (state IN ('active','archived','superseded')),
  thread_id     uuid,
  supersedes    uuid REFERENCES memory.facts ON DELETE SET NULL,
  valid_from    timestamptz,
  valid_to      timestamptz,
  source        text NOT NULL CHECK (source IN ('extractor','explicit','mcp','compaction','import','reflection')),
  derived_by    text,
  confidence    real NOT NULL DEFAULT 1.0 CHECK (confidence BETWEEN 0 AND 1),
  created_at    timestamptz NOT NULL DEFAULT now(),
  updated_at    timestamptz NOT NULL DEFAULT now(),
  last_accessed timestamptz,
  access_count  integer NOT NULL DEFAULT 0,
  strength      real NOT NULL DEFAULT 1.0,
  -- Outside content (email, web) may never become a standing instruction.
  CHECK (NOT (kind = 'instruction' AND trust = 'outside'))
);
CREATE INDEX ON memory.facts (username, state);
CREATE INDEX ON memory.facts USING gin (topics);
CREATE INDEX ON memory.facts USING gin (fts);
CREATE INDEX ON memory.facts USING hnsw (embedding vector_cosine_ops) WHERE state = 'active';
CREATE INDEX ON memory.facts (thread_id) WHERE thread_id IS NOT NULL;

CREATE TABLE memory.fact_sources (
  fact_id    uuid NOT NULL REFERENCES memory.facts ON DELETE CASCADE,
  episode_id uuid NOT NULL,
  episode_at timestamptz NOT NULL,
  PRIMARY KEY (fact_id, episode_id),
  -- Forgetting an episode removes the provenance link; the forget code then
  -- deletes or regenerates facts left without sources.
  FOREIGN KEY (episode_id, episode_at) REFERENCES memory.episodes (id, occurred_at)
    ON DELETE CASCADE
);
CREATE INDEX ON memory.fact_sources (episode_id);

CREATE TABLE memory.fact_links (
  src uuid NOT NULL REFERENCES memory.facts ON DELETE CASCADE,
  dst uuid NOT NULL REFERENCES memory.facts ON DELETE CASCADE,
  rel text NOT NULL CHECK (rel IN ('related','contradicts','elaborates','derived_from','merged_into')),
  PRIMARY KEY (src, dst, rel)
);

CREATE TABLE memory.topics (
  name      text PRIMARY KEY CHECK (name ~ '^[a-z0-9][a-z0-9 &-]{0,47}$'),
  aliases   text[] NOT NULL DEFAULT '{}',
  embedding vector(384),
  use_count integer NOT NULL DEFAULT 0
);

-- Audit. Deletes record that something was deleted, never what it said.
CREATE TABLE memory.events (
  id        bigserial PRIMARY KEY,
  at        timestamptz NOT NULL DEFAULT now(),
  object_id uuid,
  username  text,
  op        text NOT NULL,
  actor     text NOT NULL,
  detail    jsonb
);
CREATE INDEX ON memory.events (at);

CREATE TABLE memory.health_samples (
  at      timestamptz PRIMARY KEY DEFAULT now(),
  metrics jsonb NOT NULL
);

CREATE TABLE memory.jobs (
  id         bigserial PRIMARY KEY,
  kind       text NOT NULL,
  payload    jsonb NOT NULL DEFAULT '{}',
  state      text NOT NULL DEFAULT 'queued' CHECK (state IN ('queued','running','done','failed')),
  attempts   integer NOT NULL DEFAULT 0,
  last_error text,
  not_before timestamptz NOT NULL DEFAULT now(),
  dedupe_key text,
  created_at timestamptz NOT NULL DEFAULT now(),
  updated_at timestamptz NOT NULL DEFAULT now()
);
CREATE INDEX ON memory.jobs (state, not_before);
CREATE UNIQUE INDEX jobs_dedupe ON memory.jobs (dedupe_key) WHERE state IN ('queued','running');

-- Row-level security. FORCE applies it to the owner too, so a code path that
-- forgets to set a person sees nothing rather than everything.
ALTER TABLE memory.episodes ENABLE ROW LEVEL SECURITY;
ALTER TABLE memory.episodes FORCE ROW LEVEL SECURITY;
ALTER TABLE memory.facts ENABLE ROW LEVEL SECURITY;
ALTER TABLE memory.facts FORCE ROW LEVEL SECURITY;
CREATE POLICY person ON memory.episodes USING (memory.can_see(username))
  WITH CHECK (memory.can_see(username));
CREATE POLICY person ON memory.facts USING (memory.can_see(username))
  WITH CHECK (memory.can_see(username));

GRANT USAGE ON SCHEMA memory TO homeai_app, homeai_mcp, homeai_ro;
GRANT SELECT ON ALL TABLES IN SCHEMA memory TO homeai_ro;
GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA memory TO homeai_app;
GRANT USAGE ON ALL SEQUENCES IN SCHEMA memory TO homeai_app;
GRANT SELECT ON memory.people, memory.identities, memory.topics, memory.health_samples
  TO homeai_mcp;
GRANT SELECT, INSERT, UPDATE, DELETE ON memory.episodes, memory.facts, memory.fact_sources,
  memory.fact_links TO homeai_mcp;
GRANT INSERT ON memory.events, memory.jobs TO homeai_mcp;
GRANT USAGE ON ALL SEQUENCES IN SCHEMA memory TO homeai_mcp;
GRANT EXECUTE ON ALL FUNCTIONS IN SCHEMA memory TO homeai_app, homeai_mcp, homeai_ro;

-- Let the migrating user (the only login) take each role with SET ROLE.
DO $$
BEGIN
  EXECUTE format('GRANT homeai_app, homeai_mcp, homeai_ro TO %I WITH SET TRUE', current_user);
END $$;
