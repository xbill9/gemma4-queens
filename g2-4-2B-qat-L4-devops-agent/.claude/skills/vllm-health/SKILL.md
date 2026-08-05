---
name: vllm-health
description: End-to-end health check of the vLLM Gemma 4 service on the GCE L4 VM — auth, instance state, port reachability, model listing, and a latency probe. Use when asked whether the vLLM/Gemma endpoint is up or when a tool call to it fails.
---

Diagnose the live vLLM deployment layer by layer, stopping at the first one that fails and reporting
which layer broke. Run the steps in order.

## 1. Auth

```bash
gcloud auth application-default print-access-token >/dev/null 2>&1 && echo "ADC ok" || echo "ADC MISSING"
gcloud config get-value project
```

If ADC is missing, tell the user to run `source ./set_adc.sh` (it is interactive — don't try to drive
the browser login yourself) and stop.

## 2. Instance state

```bash
make status
```

Read `status` and `networkInterfaces[0].accessConfigs[0].natIP`. Distinguish the cases:

- **No such instance** — it was never deployed or was destroyed. `make deploy` (or `make deploy-vllm-spot`).
- **`TERMINATED`** — if this was deployed with `deploy-vllm-spot`, the most likely cause is **spot
  preemption**, and the instance is gone for good (`--instance-termination-action=DELETE`). Otherwise
  it was stopped; `gce_start` brings it back, but note the external IP changes on restart.
- **`RUNNING` but no natIP** — the VM has no external address; nothing downstream will reach it.

## 3. Port reachability

```bash
IP=$(make -s endpoint)
curl -s -m 10 -o /dev/null -w '%{http_code} %{time_total}s\n' "http://$IP:8080/health"
```

There is no auth on this endpoint — the VM serves 8080 directly. So a hang or connection refusal means
one of two things, and they are worth separating:

- **Firewall** — confirm the rule exists: `gcloud compute firewall-rules describe allow-vllm-8080`.
  `make deploy` creates it with a leading `-`, so its failure is silently ignored during deploy.
- **Container not up yet** — the VM's startup script installs Docker and pulls `vllm/vllm-openai:nightly`
  before serving. On a fresh `make deploy` this takes many minutes. Check with:
  ```bash
  gcloud compute instances get-serial-port-output gpu-2b-qat-l4-devops-agent --zone=us-east4-a | tail -50
  ```
  (instance name and zone are `SERVICE_NAME` / `ZONE` in the `Makefile` — read them there if overridden)
  or use the `gce_analyze_gpu_logs` MCP tool.

## 4. Model listing

```bash
curl -s "http://$IP:8080/v1/models" | python3 -m json.tool
```

Note the returned model `id` — it is what vLLM actually loaded. If it differs from `MODEL_NAME` in the
environment or `.agents/mcp_config.json`, that mismatch is a real finding: the code prefers the live
value via `get_active_model_name()`, but any hand-written curl or `make query` using `MODEL_PATH` will 404.

An empty model list with a healthy port usually means vLLM failed to load the checkpoint — check the
serial log for a `compressed-tensors` / quantization error or an HF auth failure (the startup script
pulls `hf-token` from Secret Manager; if that secret is missing the download 401s).

## 5. Latency probe

```bash
make query PROMPT="Reply with the single word: ok"
```

Report time-to-response. If the reply is garbled rather than slow, suspect `--dtype` — this checkpoint
requires `bfloat16`, and fp16 produces overflow garbage.

## Report

Give a short verdict: which layer failed (or all green), the observed IP, the live model id, and the
probe latency. If something failed, give the one concrete next action — don't list every possibility.
