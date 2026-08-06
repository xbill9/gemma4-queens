---
name: inf-status
description: Report what Inferentia/inf2 infrastructure is currently running for this project and whether it is costing money. Use when the user asks "what's running", "am I paying for anything", "is the endpoint up", or before provisioning anything new.
---

Read-only sweep. Never provision, restart, or destroy anything from this skill.

## Steps

1. **Determine the region first.** Do not assume — this project has no single authoritative region.
   Read `active_deployment_region.txt`, and note that `Makefile`/`server.py` default to `us-east-1`
   while `.env`/`.mcp.json` set `us-west-2`. State which region you checked in your output.

2. Call `mcp__inf-devops-agent__status_ec2` — the instance layer. This is what bills.

3. Call `mcp__inf-devops-agent__status_vllm` — the container layer.

4. Call `mcp__inf-devops-agent__check_vllm` — endpoint reachability.

5. If an endpoint is up and the user wants a functional check, call
   `mcp__inf-devops-agent__verify_model_health`. Skip otherwise; it costs tokens on the served model.

## Reporting

Lead with the billing answer, because that's the real question:

- **Instance running** → say so plainly, name the instance type and how to stop it
  (`stop_ec2` — note that `destroy_vllm` / `make destroy` removes only the container and does
  **not** stop billing).
- **Instance stopped** → not billing compute, but EBS volumes persist (nothing in this repo calls
  `terminate_instances`), so storage still accrues.
- **Nothing found** → say which region you searched, and suggest checking the other one before
  concluding nothing exists.

Flag it prominently if you find **more than one running instance for the same `service_name` in one
region** — that violates this project's single-host rule and is usually an accident.
