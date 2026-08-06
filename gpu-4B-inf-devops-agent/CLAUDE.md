# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

An MCP server (`server.py`, FastMCP, ~2100 lines, 30 tools) that acts as an SRE/DevOps agent for
self-hosting **Gemma 4 (`google/gemma-4-E4B-it`) on AWS Inferentia2 (`inf2`) EC2 instances**.

**The directory name lies.** `gpu-4B-inf-devops-agent` is AWS Inferentia/Neuron — not GPU, not GCP.
Ignore the vestigial `GOOGLE_CLOUD_PROJECT` in `.mcp.json` (`server.py` never reads it) and
`init.sh` (a leftover gcloud bootstrap with no bearing on this project).

The MCP server is registered as **`inf-devops-agent`** — the `inf` prefix distinguishes it from the
sibling GPU/TPU agents in `/home/xbill/gemma4-queens/`. Its tools appear as `mcp__inf-devops-agent__*`.

Sibling projects are near-copies targeting different accelerators (`gpu-4B-L4-devops-agent`,
`tpu-2B-v6e1-devops-agent`, …). Don't copy fixes between them blindly — the backends differ.
The **git root is the parent directory** `/home/xbill/gemma4-queens`, not this folder.

## Commands

| Command | Notes |
|---|---|
| `make install` | `pip install -r requirements.txt`. No venv is created. |
| `make run` | `python server.py` — MCP server on stdio. |
| `make test` | `pytest`. **No test suite exists yet**; this target is a placeholder. |
| `make lint` | `ruff check .` && `ruff format --check .` && `mypy .` |
| `make auth` | Caches SSO creds to `.aws_creds`. |
| `make status` | Read-only. |
| `make deploy` | **Launches an `inf2.8xlarge` spot instance. Costs money.** |
| `make destroy` | Removes the Docker container only — see cost rules below. |

`bash test_inference.sh` curls a live endpoint at `localhost:8080/v1/chat/completions` — it's the only
end-to-end smoke test, and it needs a running deployment.

After editing `server.py`, the user must run `/mcp` to reconnect before changes take effect.

## Cost and safety rules

These are the mistakes that cost real money or real time.

- **Never run more than one EC2 instance per `service_name` per region.** Always call `status_ec2`
  before provisioning anything.
- **`destroy_vllm` / `make destroy` does NOT stop the instance** — it only removes the Docker
  container. You are still billed. Call `stop_ec2` to actually stop paying. Nothing in this repo
  calls `terminate_instances`, so EBS volumes persist after a stop.
- **Never `docker restart` a failed Neuron container.** A *failed* compilation is cached and replayed
  forever (manifests as `SIGHUP`). Remove the container, then
  `sudo rm -rf /var/tmp/neuron-compile-cache /home/ubuntu/.cache/neuron/*`. `deploy.sh` does this —
  it is meant to run **on the EC2 host**, not locally.
- `inf2.xlarge` (16 GB) OOMs on the ~14.5 GB neff-load peak — use the `:slim-devprefill` image
  there and `:latest` on `inf2.8xlarge`.
- **Verify the image tag exists before deploying.** The only real tags on Docker Hub are
  `latest`, `slim-devprefill`, `tp2-devprefill-512`, `tp2-2048`. A bad tag is a silent failure:
  the pull 404s, cloud-init exits with **no container**, and the instance bills indefinitely while
  looking `running` and healthy. `check_vllm` reporting `no gemma-* container` after cloud-init has
  finished means a dead deploy, not a slow one — check `aws ec2 get-console-output` immediately
  rather than waiting on `/health`.

## Region

There is **no single authoritative region** — it is chosen per deployment. The repo disagrees with
itself on purpose-ish: `Makefile`/`server.py` default to `us-east-1`, `.env`/`.mcp.json` set
`us-west-2`. Before acting on any deployment, check `active_deployment_region.txt` or call
`status_ec2` rather than assuming. Note `us-east-1a` is explicitly skipped (no inf2.xlarge capacity).

## AWS credentials

`mcp-run.sh` unsets `AWS_ACCESS_KEY_ID`/`SECRET`/`SESSION_TOKEN` and sets `AWS_PROFILE=gemma-mcp`
(a `credential_process` profile that refreshes on expiry). This is the fix for recurring
`RequestExpired` errors: the server would otherwise inherit short-lived env creds from Claude Code's
launch environment, and boto3 ranks env creds above profiles. `server.py` additionally monkey-patches
`boto3.client` to re-read `.aws_creds`. Don't "simplify" either mechanism.

**Never read, print, or commit `.aws_creds*`** — they hold live credentials and are gitignored.

## Code conventions

- **Log to stderr only** (`logging.basicConfig(stream=sys.stderr, ...)`). **Never `print()`** in
  server code — stdout is the MCP stdio channel and any write to it corrupts the protocol.
- Every `@mcp.tool()` returns a **Markdown string** with emoji section headers (`### 🚀 …`), not
  structured JSON. Match that style in new tools.
- Errors are **caught and returned as human-readable strings**, not raised — a raised exception
  surfaces to the user as an opaque MCP failure.
- Line length is **120** (`ruff.toml`), not ruff's default 88.

## Out of scope

`dev-to-*.md` and `article-notes.md` are article drafts — archive. Don't edit them as part of
unrelated work.

## Deep Neuron/vLLM reference

For the Gemma 4 cross-layer KV-sharing and hybrid 256/512 `head_dim` attention issues, the Option-B
traced-server port, and compiler/scratchpad flags, read these on demand rather than assuming:

@.agents/AGENTS.md
