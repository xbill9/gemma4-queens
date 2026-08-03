# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A single-file MCP server (`server.py`, FastMCP) that acts as a devops agent for serving Gemma 4
(`google/gemma-4-E2B-it`) with vLLM on a Google Cloud TPU v6e-1 Flex-start Queued Resource. Its tools shell
out to `gcloud` and talk HTTP to the vLLM OpenAI-compatible endpoint on port 8000. This rig is used for
**live demos** — prefer changes that keep the demo working over broad refactors.

## Commands

```
make install    # pip install -r requirements.txt
make run        # python server.py (stdio MCP server)
make test       # python test_agent.py — unittest, NOT pytest
make lint       # ruff check . && ruff format --check . && mypy .
make format     # apply ruff formatting and autofixes
make tools      # regenerate GemmaTools.md from the @mcp.tool() decorators
make benchmark  # discovers the TPU IP, then runs benchmarking_suite.py against it
make query PROMPT="..."
```

`make lint` only *checks* formatting — `make format` is what writes it. Both `make lint` and `make test`
currently pass clean; keep them that way.

## Style

- ruff is both linter and formatter; no black. `line-length = 120`, but `E501` is in the ignore list, so the
  formatter enforces width and the linter does not.
- Lint rules are `E, F, B, I` — import sorting comes from ruff's `I`, not a separate isort.
- mypy is deliberately non-strict: `check_untyped_defs = true` but `attr-defined` is globally disabled.
- Existing code uses `Optional[str]` from `typing`, not `X | None` (target is py310).
- Every subprocess call goes through `run_command(cmd: list[str])` — list args via
  `asyncio.create_subprocess_exec`, never `shell=True`. Keep it that way.
- MCP tools are `async def` and return markdown strings with emoji status prefixes (`✅`, `❌`, `📡`).

## Tool catalog is generated — don't hand-edit it

`GemmaTools.md` and the `get_help` tool both build their tool list from `mcp.list_tools()`, so they cannot
drift from the `@mcp.tool()` decorators. After adding or removing a tool, run `make tools` to refresh the
doc. `README.md` intentionally lists only a handful of highlights and points at `GemmaTools.md` for the rest.

Source of truth either way: `grep -n "^@mcp.tool" server.py`.

## Gotchas

**`startup_script_template.sh` is consumed by `str.format()`.** Placeholders are `{project_id}`, `{zone}`,
`{model_name}`, `{hf_token}`, `{tensor_parallel_size}`, `{limit_mm_per_prompt_env}`. Any other literal `{` or
`}` added to that bash file — a shell brace expansion, a `${VAR}`, a JSON literal — raises at format time and
breaks the deploy. Escape as `{{` / `}}`.

**`tpu_zones_status.md` is mutable state, not documentation.** `find_tpu` rewrites it in place to record which
zones have failed, and reads it back to skip known-bad zones. Do not hand-edit it as if it were docs.

**Endpoint discovery is dynamic.** `discover_vllm_url()` lists queued resources in `ZONE`, takes the first
`ACTIVE` one, resolves its node and external IP, and builds `http://{ip}:8000`. Never hardcode an endpoint —
the IP changes every time the Queued Resource is recreated. Use `get_vllm_endpoint` or `make endpoint`.

**Zone varies per demo.** `ZONE` / `REGION` default to `europe-west4-a` / `europe-west4` but now read
`GOOGLE_CLOUD_ZONE` / `GOOGLE_CLOUD_REGION` from the environment; the `Makefile` defaults match. Check what is
actually running before assuming — `list_queued_resources` only looks in the configured zone.

**`--tensor-parallel-size` is 1.** v6e-1 is a single chip. If you see `4` anywhere, it's copy-paste from a
larger topology.

**Don't destroy a queued resource unless asked.** Teardown is not part of routine debugging, and Flex-start
capacity can take up to 2 hours to come back.

## Auth and env

Requires both `gcloud auth login` (for the `gcloud` subprocess calls) and `gcloud auth application-default
login` (ADC, for the `google-cloud-secret-manager` client). `set_env.sh` must be **sourced**, not executed.
`init.sh` is a one-time bootstrap that blocks on `read` in its error path — don't run it non-interactively.

Env vars `server.py` reads: `GOOGLE_CLOUD_PROJECT`, `GOOGLE_CLOUD_ZONE`, `GOOGLE_CLOUD_REGION`, `MODEL_NAME`,
`ACCELERATOR_TYPE`, `TENSOR_PARALLEL_SIZE`, `LOCAL_DOCKER_IMAGE`. The HF token lives in GCP Secret Manager
under the secret id `hf-token` — never log, return, or commit it.

## Tests

`test_agent.py` mocks the whole `mcp` module and the Google Cloud clients before importing `server`. Keep unit
tests offline: mock the cloud, subprocess, and network boundaries rather than reaching out. Because `mcp` is a
`MagicMock`, anything calling `mcp.list_tools()` needs an explicit `AsyncMock` patch — see `test_get_help`.

## Git

The git root is the **parent** directory, `/home/xbill/gemma4-queens` — this project is one subdirectory of
it, alongside sibling agent projects and a `gemma-skills` submodule. `git add .` from here stages only this
subdirectory; run git commands from the repo root when you mean the whole tree.

Committed benchmark artifacts (`*.png` plots, `benchmark_results.*`, `grid_benchmark_results.csv`) are
intentionally tracked. Don't regenerate or delete them unless asked.

`AGENTS.md` in this directory is maintained by a different tool and overlaps with this file — if you change a
convention here, check whether it needs the same change there.
