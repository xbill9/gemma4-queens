# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

An MCP server (`server.py`, MCPServer) exposing SRE/DevOps tools for a self-hosted Gemma 4 vLLM
service running on a **GCE VM** (`g2-standard-4`, one NVIDIA L4). The model is the QAT INT4
checkpoint `google/gemma-4-E2B-it-qat-w4a16-ct`. Nearly every tool either shells out to `gcloud`
or calls the vLLM endpoint over the OpenAI-compatible API.

## Commands

- `make lint` — `ruff check` + `ruff format --check` + `mypy`. Not wired into `make test`, and no CI
  workflow covers this directory (`.github/workflows/` has only the Cloud Run sibling) — run it
  yourself before committing. There is no write-mode target; run `ruff format .` by hand.
- `make test` — `python test_agent.py`. This is **`unittest`**, not pytest (`unittest.IsolatedAsyncioTestCase`).
  Single test: `python -m unittest test_agent.TestDevOpsAgent.test_query_gemma4`.
  Tests mock `run_gcloud`, `subprocess.run`, and the OpenAI/httpx clients — they need no GCP credentials.
- `make run` — starts the MCP server on stdio.
- `make install` — `pip install -r requirements.txt`. `make clean` — removes `__pycache__`,
  `.ruff_cache`, `.mypy_cache`.
- `make deploy` / `make deploy-vllm-spot` / `make destroy` / `make status` / `make endpoint` /
  `make query PROMPT="..."` — act on the live GCE VM. `deploy` provisions a real L4 GPU and creates
  firewall rule `allow-vllm-8080`.
- `./init.sh` — one-time bootstrap (enables APIs, grants IAM roles). `source ./set_adc.sh` refreshes auth.

`test_logging.py` is not a unit test — it hits live Cloud Logging and is not run by `make test`.

## Working in this directory

This directory is one of several sibling agent projects under `gemma4-queens/` (GPU, TPU, Cloud Run,
and EC2 variants — `ls ..` for the current set). They are independent — do not edit siblings. **The git root is the
parent**, so `git status`/`git diff` show every sibling; never `git add -A`, stage this directory's
paths explicitly.

`GEMINI.md` is a parallel context file for the Gemini CLI. When conventions, config, or the tool list
change here, update `GEMINI.md` too.

Write for **Python 3.13** — the installed runtime, and what `pyproject.toml` pins ruff/mypy to.
The interpreter is pyenv's 3.13.14, which `.mcp.json` pins by absolute path. **Never create a
virtualenv** here; install into user site-packages (`pip install -r requirements.txt`). The `.venv/`
entry in `.gitignore` is defensive, not an invitation.

## vLLM configuration

The `deploy-vllm` target in the `Makefile` is the authoritative vLLM config. `server.py`
(`gce_deploy_vllm`, `gce_get_vllm_deployment_config`, the `config://vllm-deployment-template`
resource) and `GEMINI.md` now match it flag for flag; keep all five in sync. `DEPLOY.md` records an
older **EC2** deploy — do not copy values out of it.

Flags that are load-bearing on this checkpoint:

- `--quantization compressed-tensors` — required by the `-ct` QAT checkpoint.
- `--dtype bfloat16` — fp16 overflows and produces garbled output.
- `--tool-call-parser gemma4` **and** `--reasoning-parser gemma4` — dropping either silently breaks tool use.

This checkpoint needs a vLLM **nightly** image (`vllm/vllm-openai:nightly`).

## Gotchas

- **Never write to stdout.** Logging is stderr-only because stdout is the MCP stdio channel; a stray
  `print()` corrupts the protocol.
- Most failures are auth, not code. If a tool errors, check ADC first: `source ./set_adc.sh`.
- `.claude/settings.json` registers a `PostToolUse` hook that runs `ruff format` + `ruff check --fix`
  on every `.py` file you Write/Edit. Don't re-run the formatter by hand after edits.
- Config is env-driven, all with defaults in `server.py`: `GOOGLE_CLOUD_PROJECT`, `GOOGLE_CLOUD_LOCATION`,
  `GOOGLE_CLOUD_ZONE`, `MODEL_NAME`, `HF_TOKEN`/`HF_API_KEY` (falls back to Secret Manager secret `hf-token`).
  `VLLM_BASE_URL` is optional — when unset, `discover_vllm_url()` finds the VM's external IP via `gcloud`.
- Importing `server` has side effects: it calls `aiplatform.init(...)` and reads `.aws_creds` into `os.environ`.
- **The AWS surface is vestigial**, copied from the EC2 sibling. `DEPLOY.md` (EC2 spot), `skills-lock.json`
  (AWS agent-skills lockfile nothing reads), `.aws_creds`, and `save-aws-creds.sh` do not describe this
  project. The Makefile's GCE targets and `server.py` are the truth.
- `README.md` and `GEMINI.md` link into `server.py` with `#L…` line anchors. They are correct as of now
  but rot on the next edit — regenerate them when you move tool definitions.
- `make deploy-vllm` generates `startup_script.sh` in the repo root and deletes it at the end; an aborted
  deploy leaves it behind. It is gitignored, but delete it so the next deploy starts clean.
- **Benchmark artifacts are mostly from other projects.** Only `benchmark_report.md` is this agent
  (2B QAT on the GCE L4 VM), and it has latency/throughput matrices only — no success-rate data.
  `benchmark_report_gcp.md`, `benchmark_report_summary*.md`, `model_comparison*.md` and the charts
  describe 12B on Cloud Run or on AWS EC2. Do not cite them as this model's numbers.

## Code conventions in `server.py`

- Tools are module-level `@mcp.tool()` functions; the docstring is what the model sees — write it for the caller.
- Every tool name is prefixed `gce_` (`gce_query_gemma4`, `gce_start`, `gce_get_help`) so it is
  distinguishable from the sibling TPU/EC2 agents' tools. New tools must follow this.
- Tools return human-readable strings prefixed with a status emoji (✅ / 🔴 / ⚠️) and catch their own
  exceptions into an error string instead of raising.
- Shell out via the `run_gcloud(cmd: list[str])` helper, not `subprocess` directly — tests patch it.
- Use `get_vllm_client()` rather than constructing `AsyncOpenAI`, and `get_active_model_name(client)`
  rather than trusting `MODEL_NAME`.
- Blocking Google Cloud SDK calls go through `await asyncio.to_thread(...)`. Do not call them directly
  in an async tool.
- ruff: line-length 120, rules `E,F,B,I`, `E501` ignored. mypy runs with `check_untyped_defs`.

## Tests

Every new tool must be added to `TestDevOpsAgent.EXPECTED_TOOLS` in `test_agent.py`. `test_tools_registered`
asserts that set equals the registered tool names **exactly**, so adding or renaming a tool fails the suite
until the set is updated. `test_tool_names_are_gce_prefixed` enforces the `gce_` prefix. Prefer adding a
behavior test too.

The third-party imports at the top of `server.py` carry `# noqa: E402` because `load_dotenv(override=True)`
must run before them, so the module constants below read the `.env` values. Keep the noqa if you touch that
block — `E402` is still enforced everywhere else.
