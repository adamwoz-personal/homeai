-- 0004_auth: remote app logins (plans/REMOTE_APP_PLAN.md §4, Phase 1).
-- A separate schema the memory roles cannot read: only homeai_auth (the
-- remote gateway and bin/homeai-user) sees password hashes and device tokens.
-- Roles are a backstop, not a wall (see 0001).

DO $$
BEGIN
  IF NOT EXISTS (SELECT FROM pg_roles WHERE rolname = 'homeai_auth') THEN
    CREATE ROLE homeai_auth NOLOGIN; END IF;
END $$;

CREATE SCHEMA auth;
REVOKE ALL ON SCHEMA auth FROM PUBLIC;

CREATE TABLE auth.accounts (
  username            text PRIMARY KEY REFERENCES memory.people ON DELETE CASCADE ON UPDATE CASCADE,
  password_hash       text NOT NULL CHECK (password_hash LIKE '$argon2id$%'),
  failed_logins       int NOT NULL DEFAULT 0 CHECK (failed_logins >= 0),
  locked_until        timestamptz,
  password_changed_at timestamptz NOT NULL DEFAULT now(),
  disabled            boolean NOT NULL DEFAULT false,
  guardian_access     boolean NOT NULL DEFAULT false   -- D4: an admin may read this person's chats and images
);

CREATE TABLE auth.devices (
  device_id    text PRIMARY KEY CHECK (device_id ~ '^[0-9a-f]{32}$'),
  username     text NOT NULL REFERENCES auth.accounts ON DELETE CASCADE ON UPDATE CASCADE,
  name         text NOT NULL CHECK (length(name) BETWEEN 1 AND 64),
  token_sha256 bytea NOT NULL CHECK (length(token_sha256) = 32),
  created_at   timestamptz NOT NULL DEFAULT now(),
  last_seen    timestamptz NOT NULL DEFAULT now(),
  revoked_at   timestamptz
);
CREATE INDEX devices_username ON auth.devices (username);

-- No message content, ever: who, which device, what happened.
CREATE TABLE auth.audit (
  id        bigserial PRIMARY KEY,
  at        timestamptz NOT NULL DEFAULT now(),
  username  text,
  device_id text,
  event     text NOT NULL CHECK (event ~ '^[a-z_]{1,32}$'),
  detail    jsonb NOT NULL DEFAULT '{}'
);
CREATE INDEX audit_at ON auth.audit (at);

-- App turns are attributed in memory like any other channel: handle = device_id.
ALTER TABLE memory.identities DROP CONSTRAINT identities_channel_check;
ALTER TABLE memory.identities ADD CONSTRAINT identities_channel_check
  CHECK (channel IN ('voice','sms','email','camera','cli','mcp','app'));
ALTER TABLE memory.episodes DROP CONSTRAINT episodes_channel_check;
ALTER TABLE memory.episodes ADD CONSTRAINT episodes_channel_check
  CHECK (channel IN ('voice','sms','email','camera','cli','mcp','system','app'));

GRANT USAGE ON SCHEMA auth TO homeai_auth;
GRANT SELECT, INSERT, UPDATE, DELETE ON auth.accounts, auth.devices TO homeai_auth;
GRANT SELECT, INSERT ON auth.audit TO homeai_auth;
GRANT USAGE ON SEQUENCE auth.audit_id_seq TO homeai_auth;
GRANT USAGE ON SCHEMA memory TO homeai_auth;
GRANT SELECT ON memory.people TO homeai_auth;
GRANT SELECT, INSERT, DELETE ON memory.identities TO homeai_auth;

DO $$
BEGIN
  EXECUTE format('GRANT homeai_auth TO %I WITH SET TRUE', current_user);
END $$;
