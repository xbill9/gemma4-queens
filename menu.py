#!/usr/bin/env python3
"""Text menu for the gemma4-queens repo.

Lists the sibling Gemma 4 projects — the MCP devops agents that drive vLLM —
explains what each one targets, and shows the demos each one ships.

    ./menu.py          arrow-key menu (uses rich if it is installed)
    ./menu.py --all    print everything and exit
    ./menu.py --plain  numeric-prompt menu (no rich, no raw terminal)

Navigation: up/down (or j/k) to move, Enter to open or run, Esc/Left to go
back, q to quit. Number keys still jump straight to an entry.

./launch is the same catalog without the TUI — it cds into a project and runs
one demo, either from a picker or straight from the command line.
"""

import os
import shutil
import subprocess
import sys
import textwrap
from contextlib import contextmanager
from dataclasses import dataclass, field

ROOT = os.path.dirname(os.path.abspath(__file__))
WIDTH = min(shutil.get_terminal_size((100, 24)).columns, 100)

USE_COLOR = sys.stdout.isatty() and not os.environ.get("NO_COLOR")

try:
    from rich.console import Console, Group, RenderableType
    from rich.live import Live
    from rich.markup import escape
    from rich.panel import Panel
    from rich.rule import Rule
    from rich.table import Table
    from rich.text import Text

    HAVE_RICH = True
except ImportError:  # pragma: no cover - fallback path
    HAVE_RICH = False


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
            "The big-model TPU rig. Same single-file MCPServer devops agent shape, but serving the "
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
                "and regenerates benchmark_chart.png. The chart PNGs the markdown reports "
                "link to are not checked in — run the sweep to produce them.",
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
                "Cloud Run GPU benchmark writeup, backed by benchmark_results.csv. "
                "The benchmark_chart.png it embeds is not checked in.",
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
    (
        "tpu-jax / tpu-jax-inf2  (moved out)",
        (
            "The two pure-JAX inference engines — first-party Gemma 4 forward pass, KV cache, "
            "sampler and OpenAI-compatible server, no vLLM and no PyTorch at serving time, on "
            "TPU v6e-1 and AWS Inferentia2 — were split out of this repo and now live on their "
            "own at github.com/xbill9/tpu-jax and github.com/xbill9/tpu-jax-inf2. They are not "
            "submodules here; clone them separately."
        ),
    ),
]

COMMON = """One shape lives in this repo: operators.

  Each of the six *-devops-agent / *-agent projects is an MCP server that shells out
  to gcloud/aws, launches someone else's inference server (vllm/vllm-tpu,
  vllm/vllm-openai) in a container, and talks HTTP to it.

  The two pure-JAX inference engines that used to live here — tpu-jax and
  tpu-jax-inf2 — are now standalone repos (github.com/xbill9/tpu-jax and
  .../tpu-jax-inf2). Nothing in this repo depends on them.

Every agent project follows the same shape:

  server.py          single-file MCPServer server — the agent itself
  demo_launcher.py   the scripted "grand demo" (where present)
  Makefile           install / run / test / lint / deploy / destroy / status / endpoint / query
  README.md          requirements, env vars, tool catalog
  DEPLOY.md          deployment runbook
  CLAUDE.md          conventions and gotchas (most projects; not all)
  init.sh, set_env.sh   bootstrap and environment (source set_env.sh, do not execute it)

Typical flow:   make install  ->  make deploy  ->  make status  ->  python demo_launcher.py
The agents are MCP servers; `make run` serves them on stdio for a client to attach to."""


# ─────────────────────────────── shared helpers ───────────────────────────────


def exists(project: Project, rel: str) -> bool:
    return bool(rel) and os.path.exists(os.path.join(ROOT, project.dir, rel))


def present(project: Project) -> bool:
    return os.path.isdir(os.path.join(ROOT, project.dir))


def ready_count(project: Project) -> int:
    return sum(1 for d in project.demos if exists(project, d.path))


def rule(char: str = "─") -> str:
    return dim(char * WIDTH)


def wrap(
    text: str,
    indent: str = "  ",
    width: int | None = None,
    subsequent: str | None = None,
) -> str:
    return textwrap.fill(
        text,
        width=width or WIDTH,
        initial_indent=indent,
        subsequent_indent=subsequent if subsequent is not None else indent,
    )


def run_demo(p: Project, d: Demo, assume_yes: bool = False) -> int:
    """Confirm, then run one demo in its project directory. Cooked terminal only.

    Returns the demo's exit status, or 130 if it was skipped or interrupted.
    """
    cwd = os.path.join(ROOT, p.dir)
    print()
    print(f"  {yellow('$ cd ' + p.dir + ' && ' + d.cmd)}")
    print(
        dim("  This may talk to live cloud resources and cost money. Ctrl-C stops it.")
    )
    if not assume_yes:
        try:
            answer = input("  Run it? [y/N] ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            print()
            return 130
        if answer not in ("y", "yes"):
            print(dim("  skipped"))
            return 130
    print(rule())
    try:
        code = subprocess.run(d.cmd, shell=True, cwd=cwd, check=False).returncode
    except KeyboardInterrupt:
        print(dim("\n  interrupted"))
        code = 130
    print(rule())
    return code


# ──────────────────────────── plain (stdlib) output ───────────────────────────


def print_header() -> None:
    print()
    print(bold("  gemma4-queens — Gemma 4 on every accelerator we could get"))
    print(
        dim(
            f"  {len(PROJECTS)} projects — MCP devops agents that provision accelerators"
            " and drive vLLM on them."
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
        missing = "" if present(p) else yellow("  [absent]")
        print(f"  {bold(str(i))}. {cyan(p.dir)}{missing}")
        print(f"     {p.title}")
        print(dim(f"     {p.cloud} · {p.chip} · {ready_count(p)} demos"))
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
        mark = "" if exists(p, d.path) else dim("  [missing]")
        print(f"    {bold(str(i))}) {d.name}{mark}")
        print(f"       {yellow('$ ' + d.cmd)}")
        print(wrap(d.desc, indent="       "))
    print()
    print(rule())


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


def plain_main() -> int:
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


# ──────────────────────────────── keyboard input ──────────────────────────────

_KEYS = {
    "\r": "enter",
    "\n": "enter",
    " ": "enter",
    "\x7f": "back",
    "\x08": "back",
    "\x03": "quit",  # ctrl-c
    "\x04": "quit",  # ctrl-d
}

_ESCAPES = {
    "": "esc",
    "[A": "up",
    "[B": "down",
    "[C": "right",
    "[D": "left",
    "OA": "up",
    "OB": "down",
    "OC": "right",
    "OD": "left",
    "[H": "home",
    "[F": "end",
    "[1~": "home",
    "[4~": "end",
    "[5~": "pgup",
    "[6~": "pgdn",
}


class Keyboard:
    """Raw-mode keyboard held open for the whole menu session.

    Raw mode has to stay on between frames — in canonical mode the tty holds
    keystrokes until Enter, so a lone arrow press would never arrive.
    """

    def __init__(self) -> None:
        import termios

        self.fd = sys.stdin.fileno()
        self.termios = termios
        self.saved = termios.tcgetattr(self.fd)
        self.buf = ""

    def __enter__(self) -> "Keyboard":
        self._raw()
        return self

    def __exit__(self, *exc: object) -> None:
        self._restore()

    def _raw(self) -> None:
        import tty

        tty.setraw(self.fd)

    def _restore(self) -> None:
        self.termios.tcsetattr(self.fd, self.termios.TCSADRAIN, self.saved)

    @contextmanager
    def cooked(self):
        """Temporarily give the terminal back — for input() and subprocesses."""
        self._restore()
        try:
            yield
        finally:
            self._raw()

    def _take(self) -> "str | None":
        """Pop one complete key off the buffer, or None if it needs more bytes."""
        if not self.buf:
            return None
        ch = self.buf[0]
        if ch != "\x1b":
            self.buf = self.buf[1:]
            return _KEYS.get(ch, ch.lower())
        if len(self.buf) == 1:
            return None  # a bare Esc, or the head of a sequence still arriving
        if self.buf[1] not in "[O":
            self.buf = self.buf[1:]
            return "esc"
        for i in range(2, len(self.buf)):
            if self.buf[i].isalpha() or self.buf[i] == "~":
                seq, self.buf = self.buf[1 : i + 1], self.buf[i + 1 :]
                return _ESCAPES.get(seq, "esc")
        return None  # incomplete CSI

    def read(self) -> str:
        """Block for a single keypress and return a normalized name.

        Reads the fd directly: sys.stdin buffering would pull the tail of an
        escape sequence out of the fd, leaving select() with nothing to see.
        Key repeat delivers several sequences in one read, so exactly one key is
        consumed per call and the rest stays buffered — draining the lot would
        turn a held-down arrow into a single unrecognized (and destructive) Esc.
        """
        import select

        while True:
            key = self._take()
            if key is not None:
                return key
            if self.buf and not select.select([self.fd], [], [], 0.05)[0]:
                self.buf = self.buf[1:]  # nothing followed it — a real Esc
                return "esc"
            raw = os.read(self.fd, 64)
            if not raw:  # EOF — treat as quit rather than spinning
                return "quit"
            self.buf += raw.decode("utf-8", "ignore")


class Quit(Exception):
    """Raised from a sub-view to unwind the whole menu."""


# ─────────────────────────────── rich rendering ───────────────────────────────

MENU_ACTIONS = [
    ("common", "Common layout & conventions", "How the agents are put together"),
    ("all", "Print everything", "Dump it all to the scrollback"),
    ("quit", "Quit", "Leave the menu"),
]


def _facts(p: Project) -> Table:
    t = Table.grid(padding=(0, 2))
    t.add_column(style="bold", no_wrap=True)
    t.add_column(overflow="fold")
    t.add_row("Cloud", escape(p.cloud))
    t.add_row("Hardware", escape(p.hardware))
    t.add_row("Model", escape(p.model))
    t.add_row("Endpoint", escape(p.endpoint))
    return t


def _notes_panel(p: Project, count: int | None = None) -> Panel:
    """Gotchas. `count` shows only the first N, with a footer for the rest."""
    shown = p.notes if count is None else p.notes[:count]
    notes = Table.grid(padding=(0, 1))
    notes.add_column(width=2, no_wrap=True)
    notes.add_column(overflow="fold")
    for n in shown:
        notes.add_row(Text("•", style="yellow"), Text(n))
    if len(shown) < len(p.notes):
        notes.add_row("", Text(f"+{len(p.notes) - len(shown)} more", style="dim"))
    return Panel(notes, title="Gotchas", border_style="yellow", padding=(0, 1))


def _fit_notes(console: "Console", p: Project, budget: int) -> "Panel | None":
    """The largest gotchas panel that fits in `budget` lines, or None if none does."""
    for count in range(len(p.notes), 0, -1):
        panel = _notes_panel(p, None if count == len(p.notes) else count)
        if _height(console, panel) <= budget:
            return panel
    return None


def _window(total: int, selected: int, budget: int) -> tuple[int, int]:
    """Slice bounds over `total` rows fitting `budget` lines, keeping `selected` in view."""
    budget = max(1, budget)
    if budget >= total:
        return 0, total
    start = max(0, min(selected - budget // 2, total - budget))
    return start, start + budget


def _title_bar(subtitle: str) -> Panel:
    body = Text("gemma4-queens", style="bold")
    body.append(" — Gemma 4 on every accelerator we could get", style="bold")
    body.append("\n" + subtitle, style="dim")
    return Panel(body, border_style="cyan", padding=(0, 1))


def _footer(keys: str) -> Text:
    return Text(keys, style="dim")


def _height(console: "Console", renderable: "RenderableType") -> int:
    """How many terminal lines a renderable takes at the current width."""
    return len(
        console.render_lines(renderable, console.options.update(height=None), pad=False)
    )


def _clip(console: "Console", body: str, max_lines: int) -> Text:
    """Wrap body to the console width, cut to max_lines with an ellipsis."""
    if max_lines <= 0:
        return Text("")
    lines = textwrap.wrap(body, width=max(20, console.width))
    if len(lines) > max_lines:
        lines = lines[: max_lines - 1] + [lines[max_lines - 1][:-1] + "…"]
    return Text("\n".join(lines))


def menu_rows() -> list:
    """The top-level menu, in display order: projects, extras, then actions."""
    rows: list = [("project", i, p) for i, p in enumerate(PROJECTS)]
    rows += [("extra", len(PROJECTS) + i, e) for i, e in enumerate(EXTRAS)]
    rows += [
        ("action", len(PROJECTS) + len(EXTRAS) + i, a)
        for i, a in enumerate(MENU_ACTIONS)
    ]
    return rows


def menu_entries() -> list:
    """menu_rows() flattened for display: one line each, section heads interleaved."""
    entries: list = []
    section = None
    for row in menu_rows():
        kind, _pos, item = row
        key = item.group if kind == "project" else kind
        if key != section:
            section = key
            label = {
                "project": item.group if kind == "project" else "",
                "extra": "also in this repo",
            }.get(kind, "")
            entries.append(("head", label))
        entries.append(("row", row))
    return entries


def _ellipsize(text: str, width: int) -> str:
    """Hard-truncate to `width` columns. Rich drops whole columns from an
    over-wide grid — including the cursor — so nothing may overflow."""
    if width <= 0:
        return ""
    return text if len(text) <= width else text[: max(1, width - 1)] + "…"


def _menu_table(
    console: "Console", entries: list, index: int, top: str, bottom: str
) -> Table:
    """One line per entry — the caller has already trimmed `entries` to fit."""
    # (cursor, number, name, meta) as plain strings, so widths can be budgeted.
    cells: list = []
    for i, (kind, value) in enumerate(entries):
        if (i == 0 and top) or (i == len(entries) - 1 and bottom):
            cells.append((None, "", "", top if i == 0 else bottom, ""))
        elif kind == "head":
            cells.append((None, "", "", value, ""))
        else:
            row_kind, pos, item = value
            if row_kind == "project":
                flag = "" if present(item) else "  [absent]"
                cells.append(
                    (
                        row_kind,
                        pos,
                        str(pos + 1),
                        item.dir,
                        f"{item.chip} · {ready_count(item)} demos{flag}",
                    )
                )
            elif row_kind == "extra":
                cells.append((row_kind, pos, str(pos + 1), item[0], ""))
            else:
                key, label, hint = item
                cells.append((row_kind, pos, key[0], label, hint))

    avail = console.width - 8  # cursor + number columns and their padding
    name_w = max((len(c[3]) for c in cells), default=0)
    meta_w = max((len(c[4]) for c in cells), default=0)
    if name_w + meta_w > avail:
        meta_w = max(0, avail - name_w)
        name_w = min(name_w, avail)

    table = Table.grid(padding=(0, 1))
    table.add_column(width=2, no_wrap=True)  # cursor
    table.add_column(width=2, justify="right", no_wrap=True)  # number
    table.add_column(width=name_w, no_wrap=True)  # name
    if meta_w >= 8:
        table.add_column(width=meta_w, no_wrap=True)  # meta

    for kind, pos, number, name, meta in cells:
        if kind is None:  # a section head or a scroll marker
            cell: list = ["", "", Text(_ellipsize(name, name_w), style="dim")]
        else:
            selected = pos == index
            style = "cyan" if kind in ("project", "extra") else "none"
            if selected:
                style = "bold cyan" if kind in ("project", "extra") else "bold"
            cell = [
                "▸" if selected else " ",
                number,
                Text(_ellipsize(name, name_w), style=style),
            ]
        if meta_w >= 8:
            cell.append(Text(_ellipsize(meta, meta_w), style="dim"))
        table.add_row(*cell)
    return table


def render_menu(console: "Console", index: int, compact: bool = False) -> Group:
    rows = menu_rows()
    index = max(0, min(index, len(rows) - 1))
    entries = menu_entries()
    selected = next(
        i
        for i, (kind, value) in enumerate(entries)
        if kind == "row" and value[1] == index
    )

    title: RenderableType = (
        Text("gemma4-queens", style="bold cyan", no_wrap=True, overflow="ellipsis")
        if compact
        else _title_bar(
            f"{len(PROJECTS)} MCP devops agents that provision accelerators "
            "and drive vLLM on them."
        )
    )
    footer = _footer("  ↑↓/jk move · enter open · 1-9 jump · a print all · q quit")
    # The list and the key hints are the menu; the detail pane is a bonus. Budget
    # the list first, then spend what is left — never more than the screen holds,
    # or Live(screen=True) crops the footer off the bottom.
    budget = console.size.height - _height(console, title) - _height(console, footer)
    list_budget = max(3, min(len(entries), budget - 5))
    spare = budget - list_budget - 1  # -1 for the rule under the list
    if spare < 3:  # no room for a useful detail pane — give the lines back
        list_budget, spare = max(1, min(len(entries), budget)), 0

    start, end = _window(len(entries), selected, list_budget)
    # A marker takes the place of a row, so it can never displace the cursor.
    top = bottom = ""
    if start and selected != start:
        top = f"↑ {sum(1 for kind, _ in entries[: start + 1] if kind == 'row')} more"
    if end < len(entries) and selected != end - 1:
        bottom = f"↓ {sum(1 for kind, _ in entries[end - 1 :] if kind == 'row')} more"
    table = _menu_table(console, entries[start:end], index, top, bottom)

    parts: list = [title, table]
    if spare >= 3:
        parts += [Rule(style="dim"), _detail_for(console, rows[index], spare)]
    parts.append(footer)
    group = Group(*parts)
    if _height(console, group) <= console.size.height:
        return group
    if spare:  # the detail pane overran its budget — drop it
        group = Group(title, table, footer)
        if _height(console, group) <= console.size.height:
            return group
    # Still too tall: the boxed title bar is the last thing that can go.
    return group if compact else render_menu(console, index, compact=True)


def _detail_for(console: "Console", row, budget: int) -> Group:
    """Detail pane for the highlighted row, trimmed to `budget` lines."""
    kind, _pos, item = row
    if kind == "project":
        head = Text(item.title, style="bold")
        tail = Text(
            f"{len(item.demos)} demos ({ready_count(item)} runnable) · "
            f"{len(item.notes)} gotchas — press enter to open",
            style="green",
        )
        body = item.blurb
    elif kind == "extra":
        head = Text(item[0], style="bold cyan")
        tail = None
        body = item[1]
    else:
        head = Text(item[1], style="bold")
        tail = None
        body = item[2]

    if budget <= 1:
        return Group(head)
    if tail is not None and budget <= 3:
        return Group(head, tail)
    reserved = 2 if tail is None else 4  # head + blanks (+ tail)
    text = _clip(console, body, budget - reserved)
    parts: list = [head]
    if text.plain:
        parts.extend(["", text])
    if tail is not None:
        parts.extend(["", tail])
    return Group(*parts)


def render_project(
    console: "Console", p: Project, index: int, show_notes: bool, compact: bool = False
) -> Group:
    header = Text(p.title, style="bold")
    header.append(f"\n{p.dir}/", style="cyan")
    if not present(p):
        header.append("   [not on disk]", style="yellow")
    title: RenderableType = Panel(header, border_style="cyan", padding=(0, 1))
    footer = _footer("  ↑↓/jk move · enter run · g gotchas · esc/← back · q quit")

    # Boxed, the header and demo list cost 9 lines before any content. Below that
    # the boxes come off — a cramped view still beats one cropped by Live.
    compact = compact or console.size.height < _height(console, title) + 5
    if compact:
        title = Text(p.title, style="bold", no_wrap=True, overflow="ellipsis")

    # The demo list, its detail and the key hints always stay on screen; the
    # background (gotchas, facts, blurb) fills whatever room is left over.
    avail = console.size.height - _height(console, title) - _height(console, footer)

    # Window the demo list so the cursor is always visible on a short screen.
    list_budget = max(1, min(len(p.demos), avail - (3 if compact else 5)))
    start, end = _window(len(p.demos), index, list_budget)
    # Truncate to fit rather than letting rich shed the cursor column.
    name_w = max(4, min(max(len(d.name) for d in p.demos), console.width - 10))
    demos = Table.grid(padding=(0, 1))
    demos.add_column(width=2, no_wrap=True)
    demos.add_column(width=2, justify="right", no_wrap=True)
    demos.add_column(width=name_w, no_wrap=True)
    demos.add_column(width=1, no_wrap=True)
    for i in range(start, end):
        # As in the main menu, a scroll marker never takes the cursor's row.
        if i == start and start and index != i:
            demos.add_row(
                "", "", Text(_ellipsize(f"↑ {start + 1} more", name_w), style="dim"), ""
            )
            continue
        if i == end - 1 and end < len(p.demos) and index != i:
            more = f"↓ {len(p.demos) - end + 1} more"
            demos.add_row("", "", Text(_ellipsize(more, name_w), style="dim"), "")
            continue
        d = p.demos[i]
        selected = i == index
        ok = exists(p, d.path)
        cursor = "▸" if selected else " "
        style = "bold green" if selected else ("green" if ok else "dim")
        mark = Text("●" if ok else "○", style="green" if ok else "dim")
        demos.add_row(
            cursor, str(i + 1), Text(_ellipsize(d.name, name_w), style=style), mark
        )
    demo_panel: RenderableType = (
        demos
        if compact
        else Panel(demos, title="Demos", border_style="green", padding=(0, 1))
    )

    d = p.demos[index]
    body = d.desc
    if not exists(p, d.path):
        body += f"  ({d.path} is not present in {p.dir}/ — nothing to run.)"
    cmd_line = "$ cd " + p.dir + " && " + d.cmd
    command = Text(cmd_line, style="yellow")
    detail_budget = avail - _height(console, demo_panel)
    if _height(console, command) > max(1, detail_budget - 1):
        # A long command on a narrow screen would wrap away the whole detail pane.
        command = Text(cmd_line, style="yellow", no_wrap=True, overflow="ellipsis")
    desc = _clip(console, body, detail_budget - _height(console, command))
    detail = Group(command, desc) if desc.plain else Group(command)

    spare = avail - _height(console, demo_panel) - _height(console, detail)
    # (what to keep first when space is tight, where it goes on screen)
    candidates: list = [(1, 0, _facts(p))]
    if show_notes and p.notes:
        # Toggled on explicitly, so it outranks the facts — and it trims itself
        # to whatever room there is rather than silently not appearing.
        notes = _fit_notes(console, p, spare - 1)
        if notes is not None:
            candidates.insert(0, (0, 2, notes))
    optional: list = []
    for _priority, order, renderable in candidates:
        cost = _height(console, renderable) + 1
        if cost <= spare:
            spare -= cost
            optional.append((order, renderable))
    blurb = _clip(console, p.blurb, spare - 1)
    if blurb.plain:
        optional.append((1, blurb))

    parts: list = [title]
    for _order, renderable in sorted(optional, key=lambda item: item[0]):
        parts.extend([renderable, ""])
    parts.extend([demo_panel, detail, footer])
    group = Group(*parts)
    if _height(console, group) <= console.size.height:
        return group
    if optional:  # the background overran — the demo list is what matters
        group = Group(title, demo_panel, detail, footer)
        if _height(console, group) <= console.size.height:
            return group
    return group if compact else render_project(console, p, index, show_notes, True)


def render_project_full(p: Project) -> Group:
    """The whole project, demos expanded — for --all / 'print everything'."""
    header = Text(p.title, style="bold")
    header.append(f"\n{p.dir}/", style="cyan")
    if not present(p):
        header.append("   [not on disk]", style="yellow")

    parts: list = [
        Panel(header, border_style="cyan", padding=(0, 1)),
        _facts(p),
        "",
        Text(p.blurb),
        "",
    ]
    if p.notes:
        parts.append(_notes_panel(p))
        parts.append("")

    demos = Table.grid(padding=(0, 1))
    demos.add_column(width=2, no_wrap=True)
    demos.add_column(overflow="fold")
    for i, d in enumerate(p.demos, 1):
        entry = Text(f"{i}. {d.name}", style="bold")
        if not exists(p, d.path):
            entry.append("  [missing]", style="dim")
        entry.append("\n$ " + d.cmd, style="yellow")
        entry.append("\n" + d.desc, style="none")
        demos.add_row(Text("●" if exists(p, d.path) else "○", style="green"), entry)
        demos.add_row("", "")
    parts.append(Panel(demos, title="Demos", border_style="green", padding=(0, 1)))
    return Group(*parts)


def rich_print_all(console: "Console") -> None:
    console.print(
        _title_bar(
            f"{len(PROJECTS)} MCP devops agents that provision accelerators and drive vLLM on them."
        )
    )
    for p in PROJECTS:
        console.print(render_project_full(p))
        console.print(Rule(style="dim"))
    console.print(Text("Also in this repo", style="bold"))
    for name, desc in EXTRAS:
        console.print(Text(name, style="cyan"))
        console.print(Text(desc))
        console.print()
    console.print(Panel(Text(COMMON), title="Common layout", border_style="cyan"))


def _wait(console: "Console", prompt: str = "  Enter to continue ") -> None:
    try:
        input(prompt)
    except (EOFError, KeyboardInterrupt):
        print()


def rich_project_view(
    live: "Live", console: "Console", kb: Keyboard, p: Project
) -> None:
    index = 0
    show_notes = False
    n = len(p.demos)
    while True:
        live.update(render_project(console, p, index, show_notes), refresh=True)
        key = kb.read()
        if key == "quit" or key == "q":
            raise Quit
        if key in ("esc", "back", "left", "h"):
            return
        if key in ("up", "k"):
            index = (index - 1) % n
        elif key in ("down", "j"):
            index = (index + 1) % n
        elif key == "home":
            index = 0
        elif key == "end":
            index = n - 1
        elif key == "g":
            show_notes = not show_notes
        elif key.isdigit() and 1 <= int(key) <= n:
            index = int(key) - 1
        elif key in ("enter", "right", "l"):
            d = p.demos[index]
            live.stop()
            with kb.cooked():
                if not exists(p, d.path):
                    print(
                        dim(
                            f"\n  {d.path} is not present in {p.dir}/ — nothing to run."
                        )
                    )
                else:
                    run_demo(p, d)
                _wait(console)
            live.start(refresh=True)


def common_page() -> Group:
    return Group(
        Panel(
            Text(COMMON),
            title="Common layout & conventions",
            border_style="cyan",
            padding=(0, 1),
        ),
        _footer("  any key to go back"),
    )


def rich_main() -> int:
    console = Console(highlight=False)
    rows_len = len(menu_rows())
    index = 0
    with (
        Keyboard() as kb,
        Live(console=console, screen=True, auto_refresh=False) as live,
    ):
        while True:
            live.update(render_menu(console, index), refresh=True)
            try:
                key = kb.read()
            except (EOFError, KeyboardInterrupt):
                return 0

            if key in ("quit", "q"):
                return 0
            if key in ("up", "k"):
                index = (index - 1) % rows_len
                continue
            if key in ("down", "j"):
                index = (index + 1) % rows_len
                continue
            if key == "home":
                index = 0
                continue
            if key == "end":
                index = rows_len - 1
                continue
            if key.isdigit() and 1 <= int(key) <= len(PROJECTS) + len(EXTRAS):
                index = int(key) - 1
                continue

            action = None
            if key == "c":
                action = "common"
            elif key == "a":
                action = "all"
            elif key in ("enter", "right", "l"):
                if index < len(PROJECTS):
                    try:
                        rich_project_view(live, console, kb, PROJECTS[index])
                    except Quit:
                        return 0
                    continue
                if index < len(PROJECTS) + len(EXTRAS):
                    continue  # the detail pane already shows the whole entry
                action = MENU_ACTIONS[index - len(PROJECTS) - len(EXTRAS)][0]

            if action == "quit":
                return 0
            if action == "common":
                live.update(common_page(), refresh=True)
                if kb.read() == "quit":
                    return 0
            elif action == "all":
                live.stop()
                with kb.cooked():
                    rich_print_all(console)
                    _wait(console)
                live.start(refresh=True)


def main() -> int:
    argv = sys.argv[1:]
    if "--all" in argv or "-a" in argv:
        if HAVE_RICH:
            rich_print_all(Console(highlight=False))
        else:
            print_all()
        return 0

    interactive = sys.stdin.isatty() and sys.stdout.isatty()
    if HAVE_RICH and interactive and "--plain" not in argv:
        try:
            return rich_main()
        except KeyboardInterrupt:
            return 0
    return plain_main()


if __name__ == "__main__":
    sys.exit(main())
