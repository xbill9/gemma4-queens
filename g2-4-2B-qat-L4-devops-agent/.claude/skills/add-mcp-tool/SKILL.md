---
name: add-mcp-tool
description: Scaffold a new @mcp.tool() in server.py following this repo's conventions, plus its unit test in test_agent.py. Use when adding a capability to the GCE vLLM DevOps MCP agent.
---

Add a new MCP tool named `$ARGUMENTS` (ask for the name and what it should do if not given).

## 1. Decide the tool's shape

Classify it, because each kind has an established pattern in `server.py`:

- **gcloud tool** (deploy, start/stop, destroy, status, quotas, logs) — `async def`, calls
  `code, stdout, stderr = await run_gcloud([...])`. Pass the argv **without** the leading `"gcloud"`
  and always include `f"--project={PROJECT_ID}"` and `f"--zone={zone}"`. Model on `gce_stop` / `gce_status_vllm`.
  Do not call `subprocess.run` directly in a new tool — tests patch `server.run_gcloud`.
- **Google Cloud SDK tool** (Secret Manager, Storage, Logging, Vertex) — `async def`, every blocking
  client call wrapped in `await asyncio.to_thread(client.method, request={...})`. Model on
  `gce_save_hf_token` / `gce_analyze_cloud_logging`.
- **Inference tool** (asks the model something) — `async def`, get the client from `get_vllm_client()`
  and the model id from `get_active_model_name(client)`. Model on `gce_query_gemma4` / `gce_suggest_sre_remediation`.
- **Raw HTTP tool** (a vLLM path the OpenAI client doesn't cover, e.g. `/health`, `/metrics`) —
  `async with httpx.AsyncClient(timeout=...)` against `get_vllm_url()`. Model on `gce_get_metrics` /
  `gce_get_model_details`. Note `get_auth_token()` returns `""` here by design — the VM serves port 8080
  unauthenticated behind the `allow-vllm-8080` firewall rule.

Read the named reference tool before writing anything — match it rather than inventing a new style.

## 2. Write the tool

Place it near related tools in `server.py` (the file is loosely grouped: secrets → infra/deploy →
lifecycle → models → quotas → health/inference → monitoring → benchmarks → diagnostics), not appended
at the bottom.

Requirements:

- Decorate with `@mcp.tool()`.
- Name it `gce_<verb>_<object>`. Every tool in this server carries the `gce_` prefix so it is
  distinguishable from the sibling TPU/EC2 agents when both MCP servers are connected.
- The docstring is the tool description the model reads. Lead with one imperative line saying what it
  does, then an `Args:` block — every existing tool has one and the model relies on it.
- Parameters get real type hints and sensible defaults (`instance_name: str = DEFAULT_INSTANCE_NAME`,
  `zone: str = ZONE`). Use `Optional[str]`, not `str | None`, to match the file.
- Return a human-readable `str` prefixed with a status emoji: `✅` success, `🔴`/`❌` failure, `⚠️` degraded.
- Catch exceptions inside the tool and return the error string — do not let it raise out to the MCP layer.
- Never `print()`. stdout is the MCP stdio channel. Use `logger.info` / `logger.warning` (already
  configured to stderr).
- Truncate large payloads before returning (existing tools slice to ~1000 chars).

## 3. Write the test

In `test_agent.py`:

1. Add the tool name to `test_tools_registered`'s assertion list. This is required — it is the
   registration contract.
2. Add a unit test in `TestDevOpsAgent` following the neighbouring style:
   - `@patch("server.run_gcloud")` for gcloud tools. The mock returns the `(code, stdout, stderr)`
     tuple; assert on the constructed argv in `mock_run_gcloud.call_args`, not just the return string.
   - `@patch("server.get_vllm_client")` + `@patch("server.get_active_model_name")` for inference tools.
   - `@patch("server.httpx.AsyncClient")` (plus `server.get_vllm_url`, `server.get_auth_token`) for raw HTTP.
   - Async tests are plain `async def` methods (the class is `unittest.IsolatedAsyncioTestCase`).
   - Tests must not require GCP credentials or network.

## 4. Verify

```bash
make lint && make test
```

Both must pass. Fix anything ruff or mypy flags rather than adding ignores.

## 5. Update docs

Add the tool to the matching category list in `README.md` **and** `GEMINI.md` (they carry parallel tool
inventories), and to the `gce_get_help` tool's text in `server.py`, which lists tools by category.
