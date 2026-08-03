import asyncio
import csv
import os
import resource
import statistics
import time

import httpx
import matplotlib.pyplot as plt
from dotenv import load_dotenv

# Load .env file with override
load_dotenv(override=True)

# Load AWS credentials if .aws_creds exists
aws_creds_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".aws_creds")
if os.path.exists(aws_creds_path):
    with open(aws_creds_path, "r") as f:
        for line in f:
            if "=" in line:
                key, val = line.strip().split("=", 1)
                os.environ[key] = val

# Increase file descriptor limit for high concurrency
try:
    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    resource.setrlimit(resource.RLIMIT_NOFILE, (min(hard, 8192), hard))
    print(f"File descriptor limits increased to soft={min(hard, 8192)}, hard={hard}")
except Exception as e:
    print(f"Warning: Could not increase file descriptor limits: {e}")


# Helper to get service URL
def discover_vllm_url(service_name="gpu-2b-qat-l4-devops-agent"):
    if os.getenv("VLLM_BASE_URL"):
        return os.getenv("VLLM_BASE_URL")

    # Try AWS first
    if os.getenv("AWS_ACCESS_KEY_ID") or os.path.exists(os.path.expanduser("~/.aws/credentials")):
        try:
            import boto3

            ec2 = boto3.client("ec2", region_name="us-east-1")
            response = ec2.describe_instances(
                Filters=[
                    {"Name": "tag:Name", "Values": [service_name]},
                    {"Name": "instance-state-name", "Values": ["running"]},
                ]
            )
            for reservation in response.get("Reservations", []):
                for instance in reservation.get("Instances", []):
                    ip = instance.get("PublicIpAddress")
                    if ip:
                        return f"http://{ip}:8080"
        except Exception as e:
            print(f"Error discovering AWS URL: {e}")

    # Fallback to GCP GCE
    cmd = [
        "gcloud",
        "compute",
        "instances",
        "describe",
        service_name,
        "--project=aisprint-491218",
        "--zone=us-west1-a",
        "--format=value(networkInterfaces[0].accessConfigs[0].natIP)",
    ]
    import subprocess

    try:
        res = subprocess.run(cmd, capture_output=True, text=True, timeout=15)
        if res.returncode == 0:
            ip = res.stdout.strip()
            if ip:
                return f"http://{ip}:8080"
    except Exception as e:
        print(f"Error discovering GCP URL: {e}")
    return None


def get_auth_token():
    if os.getenv("AWS_ACCESS_KEY_ID") or os.path.exists(os.path.expanduser("~/.aws/credentials")):
        return ""
    import subprocess

    try:
        return (
            subprocess.check_output(["gcloud", "auth", "print-identity-token"], stderr=subprocess.DEVNULL, timeout=10)
            .decode("utf-8")
            .strip()
        )
    except Exception:
        return ""


async def tokenize_prompt(url, token, prompt_text, model_name="google/gemma-4-E2B-it-qat-w4a16-ct"):
    headers = {"Content-Type": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    async with httpx.AsyncClient() as client:
        try:
            res = await client.post(
                f"{url.rstrip('/')}/tokenize",
                json={"model": model_name, "prompt": prompt_text},
                headers=headers,
                timeout=15,
            )
            if res.status_code == 200:
                return res.json().get("count", len(prompt_text.split()))
        except Exception:
            pass
    return len(prompt_text.split())


async def get_prompt_for_size(url, token, size, model_name="google/gemma-4-E2B-it-qat-w4a16-ct"):
    word = " hello"
    guess_text = word * size
    count = await tokenize_prompt(url, token, guess_text, model_name)
    if count == size:
        return guess_text

    attempts = 0
    while count != size and attempts < 10:
        if count < size:
            guess_text += word * (size - count)
        else:
            guess_text = guess_text[: guess_text.rfind(word)]
        count = await tokenize_prompt(url, token, guess_text, model_name)
        attempts += 1
    return guess_text


async def run_sweep():
    url = discover_vllm_url()
    if not url:
        print("Error: Could not find vLLM URL")
        return
    print(f"Found vLLM Endpoint: {url}")
    token = get_auth_token()

    model_name = "google/gemma-4-E2B-it-qat-w4a16-ct"
    try:
        headers = {}
        if token:
            headers["Authorization"] = f"Bearer {token}"
        async with httpx.AsyncClient() as client:
            res = await client.get(f"{url.rstrip('/')}/v1/models", headers=headers, timeout=10)
            if res.status_code == 200:
                models = res.json().get("data", [])
                if models:
                    model_name = models[0]["id"]
                    print(f"Discovered active model ID: {model_name}")
    except Exception as e:
        print(f"Warning: Could not get active model ID: {e}")

    # Sweep dimensions requested: 4, 8, 16..16K context window and 1, 2..2048 concurrent users
    context_sizes = [4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384]
    concurrencies = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048]

    # Pre-generate and cache prompts
    print("Generating prompts for all context window sizes...")
    prompts_cache = {}
    for size in context_sizes:
        prompt_str = await get_prompt_for_size(url, token, size, model_name)
        prompts_cache[size] = prompt_str
        print(f"  Context size {size:5d} tokens generated.")

    all_results = []
    headers = {"Content-Type": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"

    async def send_req(client, sem, prompt_text):
        payload = {
            "model": model_name,
            "prompt": prompt_text,
            "max_tokens": 1,  # Minimize output tokens to focus on input prefill scaling
            "temperature": 0.0,
            "stream": False,
        }
        async with sem:
            start = time.perf_counter()
            try:
                res = await client.post(
                    f"{url.rstrip('/')}/v1/completions", json=payload, headers=headers, timeout=60.0
                )
                latency = time.perf_counter() - start
                if res.status_code == 200:
                    return {"success": True, "latency": latency}
                else:
                    return {"success": False, "error": f"HTTP {res.status_code}"}
            except Exception as e:
                latency = time.perf_counter() - start
                return {"success": False, "error": type(e).__name__, "latency": latency}

    # Warmup
    print("Warming up the model...")
    sem = asyncio.Semaphore(1)
    async with httpx.AsyncClient(limits=httpx.Limits(max_connections=10)) as client:
        await send_req(client, sem, prompts_cache[8])

    print("\nStarting 2D Grid Benchmark Sweep...")
    print(
        f"{'Context':<8} | {'Concurrency':<12} | {'Success Rate':<12} | {'Avg Latency':<12} | {'Req/s':<10} | {'Time':<8}"
    )
    print("-" * 75)

    for size in context_sizes:
        prompt_text = prompts_cache[size]
        for c in concurrencies:
            sem = asyncio.Semaphore(c)
            limits = httpx.Limits(max_connections=c + 50, max_keepalive_connections=c + 50)

            start_time = time.perf_counter()
            async with httpx.AsyncClient(limits=limits, timeout=90.0) as client:
                tasks = [send_req(client, sem, prompt_text) for _ in range(c)]
                batch_results = await asyncio.gather(*tasks)
            total_time = time.perf_counter() - start_time

            successes = [r for r in batch_results if r["success"]]
            failures = [r for r in batch_results if not r["success"]]

            success_rate = len(successes) / c
            latencies = [r["latency"] for r in successes]

            if latencies:
                avg_lat = statistics.mean(latencies)
                p95_lat = sorted(latencies)[int(len(latencies) * 0.95)] if len(latencies) > 0 else 0.0
            else:
                avg_lat, p95_lat = 0.0, 0.0

            req_per_sec = len(successes) / total_time if total_time > 0 else 0.0

            print(
                f"{size:<8d} | {c:<12d} | {success_rate:<12.1%} | {avg_lat:<11.2f}s | {req_per_sec:<10.2f} | {total_time:<7.2f}s"
            )

            all_results.append(
                {
                    "context_size": size,
                    "concurrency": c,
                    "success_rate": success_rate,
                    "avg_latency": avg_lat,
                    "p95_latency": p95_lat,
                    "req_per_sec": req_per_sec,
                    "total_time": total_time,
                    "num_success": len(successes),
                    "num_failure": len(failures),
                }
            )

    # Save CSV
    csv_file = "benchmark_sweep_results.csv"
    with open(csv_file, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=all_results[0].keys())
        writer.writeheader()
        writer.writerows(all_results)
    print(f"\nCSV results saved to {csv_file}")

    # Generate Matrices for Markdown Report
    latency_matrix: dict[int, dict[int, str]] = {size: {} for size in context_sizes}
    throughput_matrix: dict[int, dict[int, str]] = {size: {} for size in context_sizes}
    for r in all_results:
        latency_matrix[r["context_size"]][r["concurrency"]] = f"{r['avg_latency']:.2f}s"
        throughput_matrix[r["context_size"]][r["concurrency"]] = f"{r['req_per_sec']:.1f}"

    # Save Markdown report
    md_file = "benchmark_report.md"
    with open(md_file, "w") as f:
        f.write("# 📊 Gemma 4 QAT vLLM GPU 2D Grid Concurrency Benchmark Report\n\n")
        f.write(f"Generated at: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Endpoint: `{url}`\n")
        f.write(f"Model: `{model_name}` (NVIDIA L4 GPU GCE VM)\n\n")

        f.write("## 🕒 Average Latency Matrix (seconds)\n\n")
        header = "| Context \\ Users | " + " | ".join(f"{c}" for c in concurrencies) + " |\n"
        separator = "|---:|" + "|".join("---:" for _ in concurrencies) + "|\n"
        f.write(header)
        f.write(separator)
        for size in context_sizes:
            row = f"| **{size}** | " + " | ".join(latency_matrix[size][c] for c in concurrencies) + " |\n"
            f.write(row)

        f.write("\n## 🚀 Throughput Matrix (Requests per second)\n\n")
        f.write(header)
        f.write(separator)
        for size in context_sizes:
            row = f"| **{size}** | " + " | ".join(throughput_matrix[size][c] for c in concurrencies) + " |\n"
            f.write(row)

    print(f"Markdown report saved to {md_file}")

    # Plot Chart
    try:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 12))

        # Select representative context sizes for plotting
        plot_sizes = [4, 128, 1024, 8192, 16384]
        markers = ["o", "s", "d", "^", "v"]

        # Latency subplot
        for size, marker in zip(plot_sizes, markers, strict=False):
            subset = [r for r in all_results if r["context_size"] == size]
            concs = [r["concurrency"] for r in subset]
            avg_lats = [r["avg_latency"] for r in subset]
            ax1.plot(concs, avg_lats, marker=marker, label=f"Context: {size} tokens")

        ax1.set_xscale("log")
        ax1.set_xticks(concurrencies)
        ax1.get_xaxis().set_major_formatter(plt.ScalarFormatter())
        ax1.set_ylabel("Average Latency (seconds)")
        ax1.set_title("Gemma 4 QAT Concurrency Sweep: Latency vs. Concurrent Users")
        ax1.grid(True, which="both", ls="-", alpha=0.2)
        ax1.legend()

        # Throughput subplot
        for size, marker in zip(plot_sizes, markers, strict=False):
            subset = [r for r in all_results if r["context_size"] == size]
            concs = [r["concurrency"] for r in subset]
            reqs = [r["req_per_sec"] for r in subset]
            ax2.plot(concs, reqs, marker=marker, label=f"Context: {size} tokens")

        ax2.set_xscale("log")
        ax2.set_xticks(concurrencies)
        ax2.get_xaxis().set_major_formatter(plt.ScalarFormatter())
        ax2.set_xlabel("Concurrent Users")
        ax2.set_ylabel("Throughput (Requests/sec)")
        ax2.set_title("Gemma 4 QAT Concurrency Sweep: Throughput (Req/s) vs. Concurrent Users")
        ax2.grid(True, which="both", ls="-", alpha=0.2)
        ax2.legend()

        plt.tight_layout()
        plt.savefig("benchmark_chart.png", dpi=300)
        plt.close()
        print("Performance chart saved to benchmark_chart.png")
    except Exception as e:
        print(f"Could not generate plot: {e}")


if __name__ == "__main__":
    asyncio.run(run_sweep())
