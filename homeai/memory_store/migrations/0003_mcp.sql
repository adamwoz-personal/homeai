-- Phase 3: MCP memory tools run as homeai_mcp.
-- They add facts with topics (and prune topics after a forget), and
-- memory_health reads the job queue. Everything else homeai_mcp needs was
-- granted in 0001; row-level security still applies to every read and write.
GRANT INSERT, UPDATE, DELETE ON memory.topics TO homeai_mcp;
GRANT SELECT ON memory.jobs TO homeai_mcp;
