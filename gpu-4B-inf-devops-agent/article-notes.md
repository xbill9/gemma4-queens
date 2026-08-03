# Article Notes — DEV Summer Bug Smash Submission Material

Raw material for writing the challenge posts. Everything below is verified and archived under [`phase1/`](phase1/) — no claims need re-checking, just re-telling.

**Deadline:** August 23, 2026, 11:59 PM PDT · tag `#bugsmash` · official prefilled template required
**Tracks:** Clear the Lineup (the fixes) + Smash Stories (the narrative) — judged independently, enter both
**Sponsor category:** Best Use of Google AI (the suite runs on ADK + Gemini tool-calling, `gemini-2.5-flash`)

---

## The one-line pitch

> I built a benchmark to compare A2A agent performance across four languages — then discovered the benchmark itself had more bugs than the code it was measuring. One of them meant a whole language column *couldn't produce data*, and the workaround had been hiding in the repo the entire time.

## Story arc (Smash Stories skeleton)

1. **Setup.** `a2a-benchmark`: four agents (Python + Go on ADK/Gemini, Node.js, Rust) compute Mersenne primes via Lucas–Lehmer; a harness sweeps N=1–24 and charts calculation time and round-trip time.
2. **The itch.** The committed results stop at N=22. The script is named `run_benchmark_1_22.py`. Why 22? Nobody remembered.
3. **The hunt.** A code review turned up six suspect bugs. Reproducing them turned up a seventh — the best one.
4. **The reveal (Bug 7).** CPython 3.11+ silently caps `int→str` at 4,300 digits. The Python agent stringifies 2^19937−1 — 6,002 digits. At N=24 the tool raises `ValueError`, the A2A response carries the error text, and the benchmark logs `N/A`. The N=22 cutoff wasn't a choice; it was an unfixed crash worked around by shortening the run.
5. **The twist.** The stringified primes were *never used*. The tool returns only `elapsed_time`. The crashing line — and its Go twin — could simply be deleted.
6. **The pile-on.** Once the agents were up, every other suspect fell in one afternoon (see bug table).
7. **The encore (Bug 8).** Validating the fixes produced a new failure: with formatting removed from Go's timed region, small-N runs got fast enough that Go's `time.Duration` started printing **nanoseconds** (`836ns`) — a suffix the harness parser never handled, so the datapoint silently became N/A. *The fix made the code too fast for its own benchmark.*
8. **The double encore (Bug 9).** The next validation rerun dropped three Go datapoints for a different reason: the harness reused deterministic `contextId`s, ADK replayed the old session, and Gemini answered *"I already did that. Do you want to do it again?"* — without calling the tool. The benchmark's data now depended on the model's mood about repetition. Per-run contextIds fixed it. (Best possible evidence for the Bug 2 thesis: an LLM in the measurement path adds failure modes unrelated to the code under test.)
9. **The moral.** Benchmarks are production code. A benchmark that routes its own measurements through an LLM, formats away its own precision, and crashes above N=23 doesn't measure languages — it measures its own bugs. And: the failure you route around ("just run to 22") is the one that owns you later.

## The bugs (Clear the Lineup material)

| # | Bug | Where | Confirmed how | Fix |
|---|---|---|---|---|
| 7 | **Python agent crashes at count ≥ 24** — CPython's 4,300-digit `int→str` limit vs 6,002-digit prime | `benchmark-python/.../agent.py:41` | Live curl → `ValueError` in A2A response; baseline run shows Python `N/A` at N=24, only failing agent | Delete the `str()` — the list is discarded, only `elapsed_time` is returned |
| 3 | **Harness parses timing from LLM prose first** — regex `"It took X seconds"` vs actual phrasing "Calculating … took X seconds"; primary path already fails on real output, structured `elapsed_time` fallback silently saves it | `run_full_benchmark.py:56` | Captured full response: prose + structured artifact side by side | Read the structured tool artifact first; prose as last resort or delete |
| 2 | **Half the agents have an LLM in the request path** — Python/Go via Gemini tool-calling, Node/Rust regex-parse directly; the RTT chart presents them as one comparison | architecture | Median RTT: Rust 3.0ms, Node 5.7ms vs Go 1,420ms, Python 1,622ms — **~300–500× gap that is pipeline, not language** | Report direct vs LLM-brokered as separate series |
| 5 | **Precision floor breaks the log chart** — Node/Rust format elapsed as `%.2f` ms → `0.00ms` for small N → unplottable on log axis | `server.ts:122`, `main.rs:224` | Rust returns `0.00ms` at n=1; baseline table shows `0.0000 ms` at N=1 for both | Emit full precision (or µs); parse accordingly |
| 6 | **Agents report the requested count, not the computed count** — table caps at 26 exponents | `server.ts:122`, `main.rs:224` | n=100 → "Found first 100 Mersenne primes" (Node 4,451ms, Rust 2,161ms — both computed 26) | Report `primes.length` / `primes.len()` |
| 4 | **Python times with a non-monotonic clock** — `time.time()` vs monotonic clocks everywhere else | `agent.py` | By inspection | `time.perf_counter()` |
| 1 | **Decimal conversion inside the timed region** (Python/Go only) — honest measurement: **2.7ms of 6,240ms = 0.04%, immaterial** | `agent.py:41`, `main.go:89` | Timed with/without at N=26, best of 3 | Same line as Bug 7 — deleted anyway. *Do not overclaim this in the post; the honesty is part of the story* |
| 8 | **Harness can't parse Go's `ns` durations** — latent until the Bug 1 fix made Go's small-N runs sub-µs (`836ns`); datapoint silently became N/A | `run_full_benchmark.py` (`parse_go_time`) | First post-fix validation run: Go N=1 → N/A; text showed "calculated in 836ns" | Parse `ns` before the broader suffixes (PR #2, second commit) |
| 9 | **Deterministic contextId breaks reruns** — ADK keeps per-context session history; on a rerun Gemini answered *"I already did that. Do you want to do it again?"* and never called the tool. 3 of 24 Go datapoints silently dropped | `run_full_benchmark.py` (`run_agent_call`) | Second validation run: Go N=3/14/17 → N/A with refusal text captured | contextId embeds a per-run timestamp (PR #2, third commit) |

### PRs (all opened 2026-07-15 — link these in the post)

- **[PR #1](https://github.com/xbill9/a2a-benchmark/pull/1)** (headline): Bugs 7 + 1 + 4 — remove stringification in Python & Go, switch Python to `perf_counter()`; includes count=24 regression test. Before/after: N=24 goes from `N/A` to data.
- **[PR #2](https://github.com/xbill9/a2a-benchmark/pull/2)**: Bugs 3 + 5 — harness reads structured artifacts before LLM prose; Node/Rust emit 4-decimal timings.
- **[PR #3](https://github.com/xbill9/a2a-benchmark/pull/3)**: Bug 6 — truthful response counts (stacked on #2).
- **[PR #4](https://github.com/xbill9/a2a-benchmark/pull/4)**: Bug 2 — RTT charted as direct vs Gemini-brokered series, log scale, honest title (stacked on #2).
- Merge order: #1 anytime; #2 → then retarget #3 and #4 to `main`.

## Verbatim evidence (quote these)

Bug 7, live agent response at n=24 (`phase1/evidence/bug7_python_n24.json`):

```
"text": "Exceeds the limit (4300 digits) for integer string conversion;
         use sys.set_int_max_str_digits() to increase the limit"
```

Bug 3, same response containing both the fragile prose and the reliable artifact (`phase1/evidence/bug3_python_n5.json`):

```
LLM text:   "Calculating the first 5 Mersenne primes took 4.9591064453125e-05 seconds."
harness re: r"It took ([\d\.\-e]+) seconds"        <-- does not match
artifact:   "elapsed_time": 4.9591064453125e-05    <-- what should be read
```

Bug 6, agents lying about count (`phase1/evidence/bug6_*_n100.json`):

```
node: "Found first 100 Mersenne primes in 4450.78ms."   (computed 26)
rust: "Found first 100 Mersenne primes in 2160.45ms."   (computed 26)
```

Bug 1, honest measurement (`phase1/evidence/bug1_str_conversion_timing.txt`):

```
N=26, best of 3:  with str() 6240.04 ms   without 6237.35 ms
overhead: 2.69 ms (0.04% of reported time)
```

## Baseline numbers (the "before")

Full N=1–24 run, 2026-07-15, all artifacts in `phase1/evidence/baseline/`
(`benchmark_results_1_24.json`, `benchmark_stdout.log`, `prime_calculation_times_1_24.png`, `a2a_latency_times_1_24.png`).

Round-trip time by pipeline (Bug 2):

| Agent | Median RTT | Min RTT | Failures |
| :-- | --: | --: | :-- |
| Rust (direct) | 3.0 ms | 1.8 ms | 0 |
| Node.js (direct) | 5.7 ms | 2.9 ms | 0 |
| Go (via Gemini) | 1,420.4 ms | 1,227.2 ms | 0 |
| Python (via Gemini) | 1,622.4 ms | 1,395.8 ms | **1 — the Bug 7 crash at N=24** |

Calculation time highlights: N=1 → Node `0.0000 ms`, Rust `0.0000 ms` (Bug 5 visible in the repo's own table). N=24 → Rust 812.6ms, Go 1,451.5ms, Node 1,633.0ms, **Python N/A**.

Planned after-shots for the post: N=24 row fully populated (PR 1); small-N points restored on the log chart (PR 2); RTT chart split into direct vs LLM-brokered series (Bug 2 fix).

## Reproduction rig (shows rigor; also Google AI content)

- `phase1/start_agents.sh` — builds nothing, starts all four agents with health checks; auth via `GEMINI_API_KEY` (bypasses interactive gcloud), `GOOGLE_GENAI_USE_VERTEXAI=FALSE`, model `gemini-2.5-flash`
- `phase1/repro_bugs.sh` — one command reproduces bugs 1/5/6/7 and archives evidence
- `phase1/stop_agents.sh` — teardown
- Environment: Python 3.13.14, Go 1.26.3, Rust 1.96.0, Node v24.18.0, ADK 2.3.0
- War-story-worthy speed bump: ADK 2.3.0 imports `a2a.server.apps`, which moved in a2a-sdk 1.x — had to pin `a2a-sdk[http-server]==0.3.26` before the Python agent would even start. (Dependency rot: predicted as a risk in the design doc, encountered within the hour.)

## Angles per prize category

- **Clear the Lineup:** PR 1 is the star — a crash fix where the before/after is "a whole column of the dataset exists now," with the benchmark's own charts as proof.
- **Smash Stories:** the N=22 mystery → 4,300-digit reveal → "the benchmark was measuring its own bugs" arc; the 300–500× pipeline gap is the eye-catching chart.
- **Best Use of Google AI:** the suite is ADK + Gemini end to end; Bug 3 doubles as a practical lesson — *pull data from structured tool artifacts, not LLM prose* — which is genuinely useful Gemini tool-calling guidance.

## Facts to double-check before publishing

- 2^19937−1 digit count: 6,002 (verified via the ValueError). 2^23209−1: 6,987 digits.
- CPython limit introduced in 3.11 (CVE-2020-10735 mitigation), default 4,300 digits — cite the docs.
- Mersenne exponent table in all four agents has 26 entries (p=2 … 23209).
- LL test verifies known exponents, it does not search — say "verifies" not "finds" if precision matters.

## Article checklist

- [x] Drafts written: [posts/clear-the-lineup.md](posts/clear-the-lineup.md), [posts/smash-stories.md](posts/smash-stories.md) (front matter has `published: false`)
- [x] PR links for every fix — required by Clear the Lineup rules
- [x] Google AI section filled in (ADK, gemini-2.5-flash, tool-calling lessons)
- [x] AI-assistance disclosure line in both posts
- [x] Don't overclaim Bug 1 (0.04%) — omitted from posts except as the line that also crashed
- [x] **PUBLISHED 2026-07-15** (via DEV API, tagged `#bugsmash`, charts hosted on the `bugsmash-assets` branch):
  - Clear the Lineup: https://dev.to/xbill/my-benchmarks-python-column-was-na-for-a-year-cpythons-4300-digit-limit-and-eight-other-bugs-1hgk
  - Smash Stories: https://dev.to/xbill/why-did-my-benchmark-stop-at-n22-a-debugging-story-in-nine-bugs-3m2l
- [x] **PRs merged 2026-07-15** — #1, #2 direct; #3, #4 landed via roll-up [#5](https://github.com/xbill9/a2a-benchmark/pull/5) (they had merged into their stacked base). All nine fixes verified on `main` @ 331c8ce.
- [ ] Skim both live posts; edit voice as desired (dev.to posts are editable after publishing)
- [ ] Cross-check against the official submission template at dev.to/bugsmash — if it requires specific template sections, edit them in (submissions can be edited until Aug 23, 11:59 PM PDT)
- [x] Key rotation: declined — risk accepted by owner (2026-07-15)
