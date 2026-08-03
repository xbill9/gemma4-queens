#!/bin/bash
# Launch the gpu-devops-agent MCP server with AUTO-REFRESHING AWS creds.
# Root cause of the recurring RequestExpired: the server inherits short-lived static
# session-token env vars (AWS_ACCESS_KEY_ID/SECRET/SESSION_TOKEN) from Claude Code's launch;
# boto3 ranks env creds above profiles, so they expire in-process and a /mcp reconnect just
# re-inherits the same stale env. Fix: unset them, then use the gemma-mcp profile whose
# credential_process re-exports the default profile's creds on demand (refreshes on expiry).
export PATH="/usr/local/bin:/home/xbill/.pyenv/shims:$PATH"   # ensure aws + python3 resolve
unset AWS_ACCESS_KEY_ID AWS_SECRET_ACCESS_KEY AWS_SESSION_TOKEN
export AWS_PROFILE=gemma-mcp
exec python3 /home/xbill/gemma4-tips-aws/gpu-4B-inf-devops-agent/server.py
