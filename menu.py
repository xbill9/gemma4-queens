#!/usr/bin/env python3
"""Text menu for the gemma4-queens repo.

Lists the sibling Gemma 4 projects — the MCP devops agents that drive vLLM and
the two pure-JAX inference engines — explains what each one targets, and shows
the demos each one ships. Pure stdlib.

    ./menu.py          interactive menu
    ./menu.py --all    print everything and exit
"""

import os
import shutil
import subprocess
import sys
from dataclasses import dataclass, field

ROOT = os.path.dirname(os.path.abspath(__file__))
WIDTH = min(shutil.get_terminal_size((100, 24)).columns, 100)

USE_COLOR = sys.stdout.isatty() and not os.environ.get("NO_COLOR")


def c(text: str, code: str) -> str:
    return f"\033[{code}m{text}\033[0m" if USE_COLOR else text


def bold(t: str) -> str:
    return c(t, "1")


def dim(t: str) -> str:
    return c(t, "2")


def cyan(t: str) -> str:
    return c(t, "36")


def yellow(t: str) -> str:
    return c(t, "33")


def green(t: str) -> str:
    return c(t, "32")


@dataclass
class Demo:
    name: str
    cmd: str
    desc: str
    path: str = ""  # file that must exist for the demo to be real


@dataclass
class Project:
    dir: str
    title: str
    cloud: str
    chip: str  # short label for the menu line
    hardware: str
    model: str
    endpoint: str
    blurb: str
    group: str = "vLLM devops agents (MCP operators)"
    notes: list[str] = field(default_factory=list)
    demos: list[Demo] = field(default_factory=list)


ENGINES = "Pure-JAX inference engines (first-party model code)"

PROJECTS = [
    Project(
        dir="tpu-12B-v6e1-devops-agent",
        title="Gemma 4 12B on Cloud TPU v6e-1 (Trillium)",
        cloud="Google Cloud",
        chip="TPU v6e-1 (Trillium)",
        hardware=(
            "TPU v6e-1 (1 chip, 1x1, 32GB HBM), Flex-start Queued Resource via DWS, "
            "runtime v2-alpha-tpuv6e, container vllm/vllm-tpu:nightly"
        ),
        model="google/gemma-4-12B-it  (bf16, no QAT)",
        endpoint="http://<node-ip>:8000  (discovered at runtime — never hardcoded)",
        blurb=(
            "The big-model TPU rig. Same single-file FastMCP devops agent shape, but serving the "
            "dense 12B checkpoint on one Trillium chip — which makes it a memory-pressure project: "
            "bf16 weights eat ~24GB of the 32GB HBM, so the serving flags are load-bearing rather "
            "than defaults. Provisions Flex-start capacity, launches vLLM, discovers the endpoint "
            "by walking ACTIVE queued resources, then uses that self-hosted Gemma to triage its own "
            "Cloud Logging errors. Ships the full benchmark/plot corpus."
        ),
        notes=[
            (
                "--max-model-len 8192 and --gpu-memory-utilization 0.95 must stay in sync across "
                "startup_script_template.sh and manage_vllm_docker; raising either fails KV alloc."
            ),
            "The committed sweeps still go to 16384 — past --max-model-len 8192. Trim first.",
            "No QAT here: -qat-w4a16-ct is a GPU compressed-tensors path the TPU backend can't use.",
            "Queued-resource capacity can take up to 2h to come back — don't destroy casually.",
            "tpu_zones_status.md is mutable state (which zones failed), not docs.",
        ],
        demos=[
            Demo(
                "Grand demo (end-to-end)",
                "python demo_launcher.py",
                "Discovers the TPU vLLM endpoint, analyzes Cloud Logging errors through the "
                "self-hosted Gemma, then prints the TPU v6e deployment config.",
                "demo_launcher.py",
            ),
            Demo(
                "Benchmark suite",
                "make benchmark",
                "Resolves the live TPU IP, then runs benchmarking_suite.py against it "
                "(latency/throughput; results land in benchmark_results.{csv,json}).",
                "benchmarking_suite.py",
            ),
            Demo(
                "Grid benchmark + plots",
                "python run_grid_benchmark.py && python plot_grid_benchmark.py",
                "Sweeps concurrency x context, writes grid_benchmark_results.csv and renders "
                "the heatmaps/lineplots checked into the repo. Trim the 16384 column first.",
                "run_grid_benchmark.py",
            ),
            Demo(
                "Find TPU capacity",
                "python find_tpu.py",
                "Scans zones for v6e quota and provisions the Queued Resource in the first "
                "zone that accepts it.",
                "find_tpu.py",
            ),
            Demo(
                "Load test",
                "python load_test.py",
                "Concurrent request hammer against the vLLM endpoint.",
                "load_test.py",
            ),
            Demo(
                "Browser demo page",
                "xdg-open gemma4-e2b-v6e1-demo.standalone.html",
                "Self-contained HTML page — the slide-friendly version of the "
                "single-TPU-chip story.",
                "gemma4-e2b-v6e1-demo.standalone.html",
            ),
            Demo(
                "One-shot query",
                'make query PROMPT="..."',
                "Sends a single prompt to the running TPU endpoint.",
                "Makefile",
            ),
        ],
    ),
    Project(
        dir="tpu-2B-v5e1-devops-agent",
        title="Gemma 4 E2B on Cloud TPU v5e-1 (v5litepod)",
        cloud="Google Cloud",
        chip="TPU v5e-1 (v5litepod)",
        hardware=(
            "TPU v5litepod-1 (1 chip, 1x1), Flex-start Queued Resource, "
            "runtime v2-alpha-tpuv5-lite, container vllm/vllm-tpu:nightly"
        ),
        model="google/gemma-4-E2B-it  (bf16)",
        endpoint="http://<node-ip>:8000  (same ACTIVE-queued-resource discovery)",
        blurb=(
            "The small/cheap sibling of the 12B rig, on the older v5e Lite chip — a capacity and "
            "availability project rather than a memory one. Its distinguishing piece is tpu.env, a "
            "single source of truth for project/region/zone/model/accelerator consumed by "
            "server.py, mcp-run.sh, the Makefile and set_env.sh alike, plus a startup script that "
            "pulls the HF token from Secret Manager at boot instead of baking it into instance "
            "metadata. Serving flags are built in exactly one place so the deploy paths can't drift."
        ),
        notes=[
            (
                "Flex-start v5litepod-1 is only accepted in us-west4-a; other zones reject the "
                "provisioning model outright, regardless of quota."
            ),
            "gcloud spells v5e as 'v5litepod' — accelerator v5litepod-1, runtime v2-alpha-tpuv5-lite.",
            (
                "The Makefile TPU targets name tpu-2B-v5e1-devops-agent while the MCP tools default "
                "to vllm-gemma4-qr — they won't find each other. Use the MCP tools for agent deploys."
            ),
            "compare_chips.py / benchmark_tables.md still carry v6e labels copied from the sibling.",
        ],
        demos=[
            Demo(
                "Grand demo (end-to-end)",
                "python demo_launcher.py",
                "Discovers the endpoint, analyzes Cloud Logging errors through the self-hosted "
                "Gemma 4 E2B, prints the deployment config.",
                "demo_launcher.py",
            ),
            Demo(
                "Find v5e capacity",
                "python find_tpu.py",
                "Scans zones for v5e quota and provisions the Queued Resource where it's "
                "accepted; records failures in tpu_zones_status.md.",
                "find_tpu.py",
            ),
            Demo(
                "v5e-1 sweep plots",
                "python plot_sweep_v5e1.py",
                "Renders sweep_heatmap_v5e1.png / sweep_lines_v5e1.png from "
                "sweep_results_v5e1.csv — the only artifacts here that describe this chip.",
                "plot_sweep_v5e1.py",
            ),
            Demo(
                "Parse a raw sweep log",
                "python parse_sweep.py",
                "Turns a raw sweep run into the tidy sweep_results_v5e1.csv the plot "
                "script consumes.",
                "parse_sweep.py",
            ),
            Demo(
                "Run the MCP server standalone",
                "./mcp-run.sh",
                "Exports anything unset from tpu.env and launches server.py — this is what "
                "mcp_config.json invokes.",
                "mcp-run.sh",
            ),
            Demo(
                "Browser demo page",
                "xdg-open gemma4-e2b-v5e1-demo.standalone.html",
                "Self-contained 'Gemma 4 E2B on a single v5e chip' page for slides.",
                "gemma4-e2b-v5e1-demo.standalone.html",
            ),
            Demo(
                "Build PyTorch/XLA for v5e-1",
                "./build_pytorch_v5e1.sh",
                "Experimental torch-xla build path for this chip — a side experiment, "
                "not the vLLM serving path.",
                "build_pytorch_v5e1.sh",
            ),
        ],
    ),
    Project(
        dir="g2-4-2B-qat-L4-devops-agent",
        title="Gemma 4 E2B QAT (w4a16) on a GCE L4 VM",
        cloud="Google Cloud",
        chip="NVIDIA L4 on GCE",
        hardware="g2-standard-4 (4 vCPU / 16GiB) + 1x NVIDIA L4",
        model="google/gemma-4-E2B-it-qat-w4a16-ct  (compressed-tensors)",
        endpoint="http://<vm-external-ip>:8080  (auto-discovered from the GCE instance)",
        blurb=(
            "Same devops-agent idea on a plain GCE VM instead of a managed service: a startup "
            "script installs Docker and brings up vllm/vllm-openai:nightly with the quantized "
            "QAT weights. All MCP tools are prefixed gce_*. Has the widest benchmark corpus — "
            "both a local sweep and a GCP sweep, with comparison charts."
        ),
        notes=[
            "Quantized -ct weights require --quantization compressed_tensors.",
            "Use --dtype=bfloat16; FP16 overflows on Gemma 4 and garbles tool calls.",
            "make deploy-vllm-spot exists for cheap (preemptible) capacity.",
        ],
        demos=[
            Demo(
                "Grand demo (end-to-end)",
                "python demo_launcher.py",
                "Cloud Logging analysis -> OOMKilled remediation -> Vertex AI model-copy "
                "instructions -> deployment config -> Vertex model list -> MCP resource read.",
                "demo_launcher.py",
            ),
            Demo(
                "Benchmark sweep",
                "python benchmark_sweep.py",
                "Parameter sweep over the L4 endpoint; writes benchmark_sweep_results.csv "
                "and the benchmark_chart / comparison_chart PNGs.",
                "benchmark_sweep.py",
            ),
            Demo(
                "Read the results",
                "less benchmark_report_summary.md model_comparison.md",
                "Pre-baked writeups: L4 throughput/latency numbers and a model-vs-model "
                "comparison (local and _gcp variants of each).",
                "benchmark_report_summary.md",
            ),
            Demo(
                "Stage weights from Hugging Face",
                "python download_gemma_huggingface.py",
                "Pulls Gemma weights locally so they can be pushed to your GCS bucket.",
                "download_gemma_huggingface.py",
            ),
        ],
    ),
    Project(
        dir="gpu-4B-cloudrun-devops-agent",
        title="Gemma 4 E4B on Cloud Run GPU (serverless L4)",
        cloud="Google Cloud",
        chip="NVIDIA L4 (serverless)",
        hardware="Cloud Run gen2, 1x NVIDIA L4, 8 vCPU / 32GiB, weights via GCS FUSE",
        model="google/gemma-4-E4B-it",
        endpoint="https://<service>-<hash>.run.app  (auto-discovered via gcloud)",
        blurb=(
            "The serverless variant: no VM to babysit, scale-to-zero between demos, weights "
            "mounted from a GCS bucket with FUSE. MCP tools are prefixed cloudrun_*, including "
            "cloudrun_update_scaling for min/max instance tuning. This is the only project with "
            "CI — .github/workflows lints and tests it on every push."
        ),
        notes=[
            "Private Google Access must be on for the subnet if you use GCS FUSE.",
            "Cold starts are the demo risk here — pre-warm with min-instances=1.",
        ],
        demos=[
            Demo(
                "Grand demo (end-to-end)",
                "python demo_launcher.py",
                "Same six-step SRE story as the GCE agent, targeting the Cloud Run service: "
                "log analysis, remediation, model staging, deploy config, MCP resource.",
                "demo_launcher.py",
            ),
            Demo(
                "Read the results",
                "less benchmark_report.md",
                "Cloud Run GPU benchmark writeup plus benchmark_chart.png.",
                "benchmark_report.md",
            ),
            Demo(
                "Deploy / tear down",
                "make deploy   |   make destroy",
                "Provisions (or deletes) the Cloud Run GPU service running vLLM.",
                "Makefile",
            ),
        ],
    ),
    Project(
        dir="gpu-2B-L4-ec2-agent",
        title="Gemma 4 E2B QAT on AWS EC2 (g6.xlarge / L4)",
        cloud="AWS",
        chip="NVIDIA L4 on EC2",
        hardware="g6.xlarge — 1x NVIDIA L4 24GB, 8 vCPU / 32GiB, DLAMI Ubuntu 22.04",
        model="google/gemma-4-E2B-it-qat-w4a16-ct",
        endpoint="http://<ec2-public-ip>:8080  (discovered by EC2 Name tag)",
        blurb=(
            "Devops-agent SRE assistant running Gemma 4 E2B QAT on a plain AWS EC2 instance. "
            "Bridges AWS CloudWatch logs (with fallback to GCP Cloud Logging) with the vLLM inference endpoint. "
            "Uses SSM Run Command to drive the vLLM container on EC2, meaning the instance needs no open "
            "port 22 and no SSH key material."
        ),
        notes=[
            (
                "Instance profile MUST include AmazonSSMManagedInstanceCore or every "
                "container tool fails while the instance still looks healthy."
            ),
            "--gpus all --ipc=host is required; omitting --ipc=host crashes workers under load.",
            "./init.sh checks credentials, the G-instance vCPU quota, S3, and the SSM profile.",
        ],
        demos=[
            Demo(
                "Grand demo (end-to-end)",
                "python demo_launcher.py",
                "Cloud Logging analysis -> SRE remediation -> Vertex AI model-copy "
                "instructions -> deployment config -> Vertex model list -> MCP resource read.",
                "demo_launcher.py",
            ),
            Demo(
                "Benchmark sweep",
                "python benchmark_sweep.py",
                "Sweeps the EC2 endpoint; writes benchmark_sweep_results.csv, "
                "benchmark_chart.png and comparison_chart.png.",
                "benchmark_sweep.py",
            ),
            Demo(
                "Read the results",
                "less benchmark_report_summary.md model_comparison.md",
                "E2B QAT numbers on NVIDIA L4 and how it stacks up against other backends.",
                "benchmark_report_summary.md",
            ),
            Demo(
                "Lifecycle & Query",
                "make deploy   |   make status   |   make query",
                "Deploy/destroy the vLLM stack, check status, or run one-shot query via endpoint.",
                "Makefile",
            ),
        ],
    ),
    Project(
        dir="gpu-4B-inf-devops-agent",
        title="Gemma 4 E4B on AWS Inferentia2 (Neuron)",
        cloud="AWS",
        chip="AWS Inferentia2",
        hardware="inf2 (Inferentia2, 2 Neuron cores on inf2.xlarge), neuronx vLLM container",
        model="google/gemma-4-E4B-it",
        endpoint="http://<ec2-public-ip>:8080  (discovered by EC2 tags)",
        blurb=(
            "The porting project, not just a deployment. Getting Gemma 4 onto Neuron meant "
            "patching GQA into NeuronX Distributed and compiling the model graph — hence "
            "apply_gqa_patch.py, inject_gqa.py, the nxd-gemma-*-inf2-work trees (12B / 26B MoE / "
            "31B), and an upstream PR branch in aws-neuron-samples-pr. The dev-to-*.md files are "
            "the written-up story of each port."
        ),
        notes=[
            "The directory name lies: 'gpu' here means Inferentia/Neuron, and it is AWS, not GCP.",
            "MCP server registers as inf-devops-agent (tools appear as mcp__inf-devops-agent__*).",
            "No demo_launcher.py and no test suite yet — make test is a placeholder.",
        ],
        demos=[
            Demo(
                "Smoke-test inference",
                "./test_inference.sh",
                "Fires a request at the running Neuron vLLM endpoint to prove the port works.",
                "test_inference.sh",
            ),
            Demo(
                "Deploy to Inferentia",
                "./deploy.sh   (or make deploy)",
                "Provisions inf2, pulls the neuronx vLLM image, compiles and serves the model. "
                "Compilation is slow — expect a long first run.",
                "deploy.sh",
            ),
            Demo(
                "Multi-device deploy",
                "python deploy_8x.py",
                "The 8-device variant used for the larger 12B/26B/31B ports.",
                "deploy_8x.py",
            ),
            Demo(
                "Read the port writeups",
                "less dev-to-gemma4-family-inferentia-comparison.md",
                "Comparison across the whole Gemma 4 family on Inferentia, plus per-model "
                "articles (4B, 12B, 26B MoE, 31B) and GEMMA4_INFERENTIA_PORT.md.",
                "dev-to-gemma4-family-inferentia-comparison.md",
            ),
            Demo(
                "Inspect Neuron logs",
                "python get_docker_logs.py",
                "Pulls the vLLM/Neuron container logs off the instance over SSM.",
                "get_docker_logs.py",
            ),
        ],
    ),
    Project(
        dir="tpu-jax",
        group=ENGINES,
        title="Gemma 4 E2B QAT on TPU v6e-1 — pure JAX, no vLLM, no PyTorch",
        cloud="Google Cloud",
        chip="TPU v6e-1 (Trillium)",
        hardware=(
            "ct6e-standard-1t, 1 chip, 32GB HBM3; bare flex-start TPU VM running "
            "jax[tpu] directly — no Docker, no vLLM"
        ),
        model="google/gemma-4-E2B-it-qat-w4a16-ct",
        endpoint="http://<host>:8000  (served by jax_openai_server.py itself; no discovery layer)",
        blurb=(
            "Not a devops agent — a from-scratch inference engine. ports/gemma4/ reads the QAT "
            "safetensors with no PyTorch in the path and implements the Gemma 4 forward, cached "
            "decode and quantized KV in JAX; jax_engine.py wraps it as a stateful generator and "
            "jax_openai_server.py puts an OpenAI-compatible HTTP/SSE face on it. It exists because "
            "the vLLM TPU stack couldn't load this QAT export, and became a measurement project "
            "about buffer donation, INT8 KV, static shapes, and the gap between a fast kernel and "
            "a useful server."
        ),
        notes=[
            (
                "Kernel speed is not serving speed: 2,888 tok/s is a static-shape decode kernel; the "
                "real checkpoint over HTTP measured ~139 tok/s, and concurrency never batches."
            ),
            (
                "Earlier numbers were withdrawn — read benchmarks/runs/2026-07-29-kv-quant-v6e1/"
                "REPORT.md before quoting any capacity figure. plot_benchmark.py is RETRACTED DATA."
            ),
            (
                "skills/ and .claude/skills/ are generated snapshots — edit the root sources, "
                "then run make skill. .mcp.json is gitignored (embeds the GCP project id)."
            ),
            (
                "CPU tests use a synthetic checkpoint: they validate numerics and scheduling, "
                "never throughput, HBM capacity or Pallas performance."
            ),
        ],
        demos=[
            Demo(
                "Serve the pure-JAX engine",
                "python3 jax_openai_server.py --model google/gemma-4-E2B-it-qat-w4a16-ct "
                "--kv-cache-dtype int8 --quant-mode w4a16 --max-model-len 8192 --port 8000",
                "Boots the FastAPI OpenAI-compatible server on a TPU VM with INT8 KV cache; "
                "exposes chat/completions, SSE streaming, /health and /metrics.",
                "jax_openai_server.py",
            ),
            Demo(
                "Corrected kernel sweep",
                "python3 ports/gemma4/jax_e_benchmark_sweep_v2.py "
                "--batch-sizes 1,2,4,8,16,32,64 --contexts 8,128,512,2048 --json-out results.json",
                "The methodologically corrected prefill + cached-decode benchmark (jitted "
                "prefill, real KV cache, isolated processes). The v1 sweep is retracted.",
                "ports/gemma4/jax_e_benchmark_sweep_v2.py",
            ),
            Demo(
                "CPU correctness suite",
                "python3 -m unittest discover -s tests",
                "12 offline modules: KV-cache parity, chunked prefill, buffer donation, "
                "quantized KV, PLE quantization, windowed KV, OpenAI server regressions.",
                "tests",
            ),
            Demo(
                "Kernel-gap profiling",
                "python3 benchmarks/queued/kernel_gap_suite.py",
                "JAX trace tooling that measures the gap between compiled kernel time and "
                "wall-clock serving time.",
                "benchmarks/queued/kernel_gap_suite.py",
            ),
            Demo(
                "Read the corrected report",
                "less benchmarks/runs/2026-07-29-kv-quant-v6e1/REPORT.md",
                "The revalidation writeup plus the correction history — required reading "
                "before quoting any capacity number.",
                "benchmarks/runs/2026-07-29-kv-quant-v6e1/REPORT.md",
            ),
            Demo(
                "Install the TPU skill + MCP server",
                "make skill-install",
                "Regenerates the tpu-management skill snapshots and installs them to "
                "~/.claude/skills, registering the tpu-devops MCP server.",
                "refresh_skill.py",
            ),
        ],
    ),
    Project(
        dir="tpu-jax-inf2",
        group=ENGINES,
        title="Gemma 4 E2B QAT on AWS Inferentia2 — the same JAX engine, ported to Neuron",
        cloud="AWS  (TPU v6e-1 retained as the parity reference)",
        chip="AWS Inferentia2 (NeuronCore-v2)",
        hardware=(
            "EC2 inf2.xlarge / inf2.8xlarge, Ubuntu 24.04 Neuron AMI (SDK 2.31.0), "
            "jax-neuronx via the PJRT 'neuron' plugin; retained gp3 compile-cache volume"
        ),
        model="google/gemma-4-E2B-it-qat-w4a16-ct",
        endpoint="http://127.0.0.1:8000  (loopback only, systemd unit gemma4-jax-inf2; reach via SSM)",
        blurb=(
            "tpu-jax forked onto Inferentia2. The model math, loader, cached decode, OpenAI API and "
            "benchmark methodology are shared with the TPU project — only the platform layer is "
            "new: deployments/aws-inf2/ (a plan/apply EC2 launcher, user_data bootstrap and a "
            "Neuron entrypoint that verifies the backend before importing the server), jax_neuron/ "
            "(runtime probe -> compiler probe -> parity harness) and docs/neuron-jax-quirks.md. "
            "It answers the Inferentia question with a yes: correct output at ~43 tok/s."
        ),
        notes=[
            (
                "--dequant-at-load is mandatory: the in-graph W4A16 matmul miscomputes on the "
                "NeuronCore (greedy decode emits one repeated token). Host-side dequant is correct."
            ),
            (
                "A too-large gather returns zeros, not an error — zero logits -> argmax 0 -> pad id -> "
                "a clean 200 OK with zero completion tokens and nothing in the logs."
            ),
            (
                "deploy.py does not build or upload the bundle; a stale --source-uri silently serves "
                "pre-fix code. Rebuild with git archive after any engine change."
            ),
            "Never quote TPU throughput or memory numbers as Inf2 claims.",
        ],
        demos=[
            Demo(
                "Plan an Inf2 launch (read-only)",
                "python3 deployments/aws-inf2/deploy.py plan --region us-east-1 "
                "--subnet-id subnet-... --security-group-id sg-... "
                "--instance-profile-name gemma4-inf2 --source-uri s3://.../bundle.tar.gz",
                "Renders the full EC2 launch plan without touching AWS. plan is the default; "
                "--apply is required to actually act.",
                "deployments/aws-inf2/deploy.py",
            ),
            Demo(
                "Neuron runtime probe",
                "python3 jax_neuron/probe.py",
                "One-minute gate exercising driver, PJRT plugin, PATH and neuronx-cc together; "
                "asserts a neuron device is visible. Runs during bootstrap too.",
                "jax_neuron/probe.py",
            ),
            Demo(
                "Compiler probe",
                "python3 jax_neuron/compile_probe.py --tiny",
                "Asks whether neuronx-cc accepts the real engine graphs (decode, prefill, "
                "sampling) rather than a hand-written stand-in.",
                "jax_neuron/compile_probe.py",
            ),
            Demo(
                "Greedy parity vs a CPU oracle",
                'python3 jax_neuron/parity.py --local-dir "$CKPT" --reference ref.json '
                "--subject-platform neuron",
                "Greedy-decodes the same prompts through JaxGemmaEngine and through HF "
                "transformers in fp32 on CPU. This catches 'runs happily, computes garbage'.",
                "jax_neuron/parity.py",
            ),
            Demo(
                "CPU correctness suite",
                "python3 -m unittest discover -s tests",
                "18 offline modules — the TPU set plus the Inf2 scaffold, backend caps, "
                "MCP user_data rendering, JAX probe, parity harness and host dequant.",
                "tests",
            ),
            Demo(
                "Read the platform quirks",
                "less docs/neuron-jax-quirks.md",
                "The silent-zero gather, allocation limits and unsupported JAX features, all "
                "measured on inf2. Read before debugging anything here.",
                "docs/neuron-jax-quirks.md",
            ),
        ],
    ),
]

EXTRAS = [
    (
        "Makefile  (repo root)",
        (
            "Fans targets out across the sub-projects: make menu (this program), submodules, "
            "install, lint, format, test, clean. Its PROJECTS list is hand-maintained — check "
            "it still matches the directories on disk before trusting a repo-wide run."
        ),
    ),
    (
        "gemma-skills/",
        (
            "Git submodule (github.com/google-gemma/gemma-skills), not currently checked out. "
            "Two Claude/agent skills: gemma-dev (building apps with Gemma, model selection, "
            "deployment) and gemma-trainer (SFT/DPO/RLHF fine-tuning on local hardware, "
            "GGUF/LiteRT conversion). Init with: git submodule update --init"
        ),
    ),
    (
        ".github/workflows/",
        (
            "CI, currently scoped to gpu-4B-cloudrun-devops-agent only — it lints and tests "
            "that one project on pushes touching its path. The other agents have no CI."
        ),
    ),
]

COMMON = """Two shapes live in this repo.

  Operators  — the six *-devops-agent / *-agent projects. Each is an MCP server that
               shells out to gcloud/aws, launches someone else's inference server
               (vllm/vllm-tpu, vllm/vllm-openai) in a container, and talks HTTP to it.
  Engines    — tpu-jax and tpu-jax-inf2. The model forward pass, KV cache, sampler and
               HTTP server are all first-party code; no vLLM and no PyTorch at serving
               time. Same engine, two accelerators (TPU v6e-1 and AWS Inferentia2).

Every agent project follows the same shape:

  server.py          single-file FastMCP server — the agent itself
  demo_launcher.py   the scripted "grand demo" (where present)
  Makefile           install / run / test / lint / deploy / destroy / status / endpoint / query
  README.md          requirements, env vars, tool catalog
  DEPLOY.md          deployment runbook
  CLAUDE.md          conventions and gotchas for this project
  init.sh, set_env.sh   bootstrap and environment (source set_env.sh, do not execute it)

Typical flow:   make install  ->  make deploy  ->  make status  ->  python demo_launcher.py
The agents are MCP servers; `make run` serves them on stdio for a client to attach to."""


def rule(char: str = "─") -> str:
    return dim(char * WIDTH)


def wrap(
    text: str,
    indent: str = "  ",
    width: int | None = None,
    subsequent: str | None = None,
) -> str:
    import textwrap

    return textwrap.fill(
        text,
        width=width or WIDTH,
        initial_indent=indent,
        subsequent_indent=subsequent if subsequent is not None else indent,
    )


def exists(project: Project, rel: str) -> bool:
    return bool(rel) and os.path.exists(os.path.join(ROOT, project.dir, rel))


def print_header() -> None:
    print()
    print(bold("  gemma4-queens — Gemma 4 on every accelerator we could get"))
    print(
        dim(
            f"  {len(PROJECTS)} projects: MCP devops agents that drive vLLM, plus two"
            " pure-JAX engines."
        )
    )
    print(rule())


def print_menu() -> None:
    print_header()
    group = None
    for i, p in enumerate(PROJECTS, 1):
        if p.group != group:
            group = p.group
            print(f"  {dim(group)}")
        have = sum(1 for d in p.demos if exists(p, d.path))
        missing = (
            "" if os.path.isdir(os.path.join(ROOT, p.dir)) else yellow("  [absent]")
        )
        print(f"  {bold(str(i))}. {cyan(p.dir)}{missing}")
        print(f"     {p.title}")
        print(dim(f"     {p.cloud} · {p.chip} · {have} demos"))
    print()
    for key, (name, _) in enumerate(EXTRAS, len(PROJECTS) + 1):
        print(f"  {bold(str(key))}. {cyan(name)}")
    print()
    print(f"  {bold('c')}. Common layout & conventions across all projects")
    print(f"  {bold('a')}. Print everything")
    print(f"  {bold('q')}. Quit")
    print(rule())


def print_project(p: Project) -> None:
    print()
    print(rule("━"))
    print(f"  {bold(p.title)}")
    print(f"  {cyan(p.dir + '/')}")
    print(rule("━"))
    print()
    print(f"  {bold('Cloud')}     {p.cloud}")
    print(f"  {bold('Hardware')}  {p.hardware}")
    print(f"  {bold('Model')}     {p.model}")
    print(f"  {bold('Endpoint')}  {p.endpoint}")
    print()
    print(wrap(p.blurb))
    if p.notes:
        print()
        print(f"  {bold(yellow('Gotchas'))}")
        for n in p.notes:
            print(wrap(n, indent="    • ", subsequent="      "))
    print()
    print(f"  {bold(green('Demos'))}")
    for i, d in enumerate(p.demos, 1):
        ok = exists(p, d.path)
        mark = "" if ok else dim("  [missing]")
        print(f"    {bold(str(i))}) {d.name}{mark}")
        print(f"       {yellow('$ ' + d.cmd)}")
        print(wrap(d.desc, indent="       "))
    print()
    print(rule())


def run_demo(p: Project, d: Demo) -> None:
    cwd = os.path.join(ROOT, p.dir)
    print()
    print(f"  {yellow('$ ' + d.cmd)}")
    print(dim(f"  in {cwd}"))
    print(
        dim("  This may talk to live cloud resources and cost money. Ctrl-C stops it.")
    )
    if input("  Run it? [y/N] ").strip().lower() not in ("y", "yes"):
        print(dim("  skipped"))
        return
    print(rule())
    try:
        subprocess.run(d.cmd, shell=True, cwd=cwd, check=False)
    except KeyboardInterrupt:
        print(dim("\n  interrupted"))
    print(rule())


def project_loop(p: Project) -> None:
    while True:
        print_project(p)
        choice = input(
            f"  Run a demo [1-{len(p.demos)}], or Enter to go back: "
        ).strip()
        if not choice:
            return
        if choice.isdigit() and 1 <= int(choice) <= len(p.demos):
            d = p.demos[int(choice) - 1]
            if not exists(p, d.path):
                print(dim(f"  {d.path} is not present in {p.dir}/ — nothing to run."))
                input("  Enter to continue ")
                continue
            run_demo(p, d)
            input("  Enter to continue ")
        else:
            print(dim("  ?"))


def print_all() -> None:
    print_header()
    for p in PROJECTS:
        print_project(p)
    print()
    print(f"  {bold('Also in this repo')}")
    for name, desc in EXTRAS:
        print(f"\n  {cyan(name)}")
        print(wrap(desc))
    print()
    print(rule())
    print()
    for line in COMMON.splitlines():
        print("  " + line)
    print()


def main() -> int:
    if "--all" in sys.argv or "-a" in sys.argv:
        print_all()
        return 0

    while True:
        print_menu()
        try:
            choice = input("  Select: ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            print()
            return 0
        if choice in ("q", "quit", "exit"):
            return 0
        if choice == "a":
            print_all()
            input("  Enter to continue ")
        elif choice == "c":
            print()
            for line in COMMON.splitlines():
                print("  " + line)
            print()
            input("  Enter to continue ")
        elif choice.isdigit() and 1 <= int(choice) <= len(PROJECTS):
            project_loop(PROJECTS[int(choice) - 1])
        elif choice.isdigit() and len(PROJECTS) < int(choice) <= len(PROJECTS) + len(
            EXTRAS
        ):
            name, desc = EXTRAS[int(choice) - len(PROJECTS) - 1]
            print()
            print(f"  {cyan(name)}")
            print(wrap(desc))
            print()
            input("  Enter to continue ")
        else:
            print(dim("  ?"))


if __name__ == "__main__":
    sys.exit(main())
