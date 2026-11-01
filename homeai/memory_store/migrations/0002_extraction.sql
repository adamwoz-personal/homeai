-- Phase 2: fact extraction bookkeeping, and summaries that never outlive a source.

-- Episodes the extractor has read. NULL = still to do.
ALTER TABLE memory.episodes ADD COLUMN extracted_at timestamptz;
CREATE INDEX episodes_unextracted ON memory.episodes (username, occurred_at)
  WHERE extracted_at IS NULL;

-- A summary (reflection profile line) is linked to its sources with
-- fact_links (summary -> source, 'derived_from'). Deleting a source by any
-- path (voice "forget", CLI, MCP, maintenance) deletes the summaries built
-- from it, so forgotten content cannot survive in a summary, and queues a
-- rebuild of that person's profile.
CREATE FUNCTION memory.drop_derived() RETURNS trigger
LANGUAGE plpgsql AS $$
DECLARE
  n int;
BEGIN
  DELETE FROM memory.facts
   WHERE id IN (SELECT src FROM memory.fact_links WHERE dst = OLD.id AND rel = 'derived_from');
  GET DIAGNOSTICS n = ROW_COUNT;
  IF n > 0 THEN
    INSERT INTO memory.jobs (kind, payload, dedupe_key)
    VALUES ('reflect', jsonb_build_object('username', OLD.username), 'reflect:' || OLD.username)
    ON CONFLICT (dedupe_key) WHERE state IN ('queued', 'running') DO NOTHING;
  END IF;
  RETURN OLD;
END $$;

CREATE TRIGGER facts_drop_derived BEFORE DELETE ON memory.facts
  FOR EACH ROW EXECUTE FUNCTION memory.drop_derived();

-- The worker reads and updates the queue; MCP may only add to it.
GRANT SELECT, UPDATE, DELETE ON memory.jobs TO homeai_app;
GRANT EXECUTE ON FUNCTION memory.drop_derived() TO homeai_app, homeai_mcp;
