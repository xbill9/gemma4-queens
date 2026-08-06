---
name: inf-teardown
description: Shut down this project's Inferentia deployment in the correct order so it actually stops billing. Use when the user is done with a deployment.
disable-model-invocation: true
---

Tear down the inf2 deployment for `$ARGUMENTS` (a `service_name`; default
`inferentia-4b-devops-agent`).

**Why this skill exists:** `make destroy` and `destroy_vllm` remove only the Docker container. The
EC2 instance keeps running and keeps billing. Teardown is a two-step sequence and the second step is
the one people forget.

## Steps

1. **Find what's actually there.** Call `mcp__inf-devops-agent__status_ec2`. Determine the region
   the same way `/inf-status` does — read `active_deployment_region.txt`, don't assume. If nothing is
   running, stop here and say so; don't run destroy commands against nothing.

2. **Show the user what you're about to tear down and confirm** — instance ID, type, region,
   service name. Do not skip this. Wait for their go-ahead.

3. `mcp__inf-devops-agent__destroy_vllm` — removes the container.

4. `mcp__inf-devops-agent__stop_ec2` — **this is the step that stops billing.** Never end a teardown
   without it.

5. **Verify** with `mcp__inf-devops-agent__status_ec2` and report the final state. Do not claim the
   teardown succeeded until this confirms it.

## Report

State plainly whether compute billing has stopped. Then note the residue: a stopped instance still
has EBS volumes attached, and nothing in this repo calls `terminate_instances` — so if the user wants
the storage gone too, they must terminate manually. Say this every time; it's the second most common
surprise after the container-vs-instance distinction.
