# Description

One resources server, several environments. Each `environments/blobfish_*` entry points this server
at a different published Blobfish suite through `suite_dir`. The server hosts a suite in-process:
per-rollout worlds built by the task package's own runtime, `POST /<tool>` dispatch, and `/verify`
through the package's own deterministic verifier.

| Environment | Suite | Tasks | Tools | Task packages |
|---|---|---|---|---|
| `blobfish_factorybench_100` | FactoryBench-100 | 100 | 94 | `harbor datasets download blobfishai/factorybench-100` |
| `blobfish_erpbench_100` | ERPBench-100 | 100 | 52 | `harbor datasets download blobfishai/erpbench-100-suite` |
| `blobfish_dealbench_100` | DealBench-100 | 100 | 36 | `harbor datasets download blobfishai/dealbench-100` |
| `blobfish_ledgerbench_100` | LedgerBench-100 | 100 | 66 | `harbor datasets download blobfishai/ledgerbench-100` |
| `blobfish_counselbench_100` | CounselBench-100 | 100 | 18 | `harbor datasets download blobfishai/counselbench-100` |
| `blobfish_salesbench_100` | SalesBench-100 | 100 | 35 | `harbor datasets download blobfishai/salesbench-100` |
| `blobfish_devopsbench_100` | DevOpsBench-100 | 100 | 97 | `harbor datasets download blobfishai/devopsbench-100` |

Task packages are executable code and are pinned by SHA-256 (`suite_manifest.json` next to each
environment); the server refuses packages that differ from the reviewed ones.

Reward modes (`reward_mode` in the server config): `fraction` (criterion fraction, the default),
`strict` (all criteria), and `gated` (fraction times the earned share of the suite's outcome
families, measured per suite by perturbing each oracle write: a wrong final write then keeps 0.30
of the reward on FactoryBench instead of 0.66). Use `gated` for RL so process without outcomes is
not rewarded; `strict_pass`, `criterion_fraction` and `outcome_fraction` are always reported.

# Example usage

See any `environments/blobfish_*/README.md` for the run commands. The default config in
`configs/blobfish_enterprise_ops.yaml` serves the first suite.

## Tests

```bash
pytest resources_servers/blobfish_enterprise_ops/tests
gym env test blobfish_factorybench_100   # onboarding verifier cases: full, zero, malformed, determinism
```

`tests/fixtures/mini_suite` is a self-contained task package, so the seed, call and verify path and
the verifier fixture (`tests/verifier_cases.jsonl`) run in CI without any download; the downloaded
suites are exercised when `BLOBFISH_<SUITE>_DIR` is set.

## Reward profile

Validation split, two repeats, temperature 0, step-capped `simple_agent`.

| Suite | Model | Rollouts | Strict passes | Mean criterion fraction | Mean tool calls | Failed calls / episode | Recorded |
|---|---|---|---|---|---|---|---|
| FactoryBench-100 | deepseek-chat (DeepSeek API) | 16 (8 tasks x 2) | 0 | 0.451 | 52.4 | 2.25 | 2026-09-06 |
| ERPBench-100 | deepseek-chat (DeepSeek API) | 28 (14 tasks x 2) | 0 | 0.423 | 43.5 | 0.54 | 2026-09-06 |
| DealBench-100 | deepseek-chat (DeepSeek API) | 24 (12 tasks x 2) | 0 | 0.090 | 64.9 | 17.54 | 2026-09-06 |
| LedgerBench-100 | deepseek-chat (DeepSeek API) | 24 (12 tasks x 2) | 0 | 0.050 | 125.3 | 52.75 | 2026-09-06 |
| CounselBench-100 | deepseek-chat (DeepSeek API) | 26 (13 tasks x 2) | 0 | 0.229 | 61.4 | 14.73 | 2026-09-06 |
| SalesBench-100 | deepseek-chat (DeepSeek API) | 22 (11 tasks x 2) | 0 | 0.219 | 69.4 | 5.27 | 2026-09-06 |
| DevOpsBench-100 | deepseek-chat (DeepSeek API) | 24 (12 tasks x 2) | 0 | 0.128 | 56.7 | 1.50 | 2026-09-06 |
| FactoryBench-100 | gemini-2.5-pro (Gemini API) | 16 (8 tasks x 2) | 0 | 0.401 | 36.6 | 2.12 | 2026-09-06 |
| ERPBench-100 | gemini-2.5-pro (Gemini API) | 28 (14 tasks x 2) | 0 | 0.511 | 44.0 | 1.96 | 2026-09-06 |
| DealBench-100 | gemini-2.5-pro (Gemini API) | 24 (12 tasks x 2) | 0 | 0.044 | 64.0 | 13.83 | 2026-09-06 |
| LedgerBench-100 | gemini-2.5-pro (Gemini API) | 24 (12 tasks x 2) | 0 | 0.050 | 19.1 | 0.92 | 2026-09-06 |
| CounselBench-100 | gemini-2.5-pro (Gemini API) | 26 (13 tasks x 2) | 0 | 0.059 | 13.5 | 0.73 | 2026-09-06 |
| SalesBench-100 | gemini-2.5-pro (Gemini API) | 22 (11 tasks x 2) | 0 | 0.139 | 35.1 | 3.82 | 2026-09-06 |
| DevOpsBench-100 | gemini-2.5-pro (Gemini API) | 24 (12 tasks x 2) | 0 | 0.120 | 45.9 | 4.25 | 2026-09-06 |
| FactoryBench-100 | nemotron-3-nano-30b-a3b (OpenRouter (nvidia/nemotron-3-nano-30b-a3b)) | 16 (8 tasks x 2) | 0 | 0.308 | 8.4 | 0.75 | 2026-09-07 |
| ERPBench-100 | nemotron-3-nano-30b-a3b (OpenRouter (nvidia/nemotron-3-nano-30b-a3b)) | 28 (14 tasks x 2) | 0 | 0.447 | 18.4 | 0.96 | 2026-09-07 |
| DealBench-100 | nemotron-3-nano-30b-a3b (OpenRouter (nvidia/nemotron-3-nano-30b-a3b)) | 24 (12 tasks x 2) | 0 | 0.068 | 27.0 | 4.96 | 2026-09-07 |
| LedgerBench-100 | nemotron-3-nano-30b-a3b (OpenRouter (nvidia/nemotron-3-nano-30b-a3b)) | 24 (12 tasks x 2) | 0 | 0.045 | 37.9 | 4.54 | 2026-09-07 |
| CounselBench-100 | nemotron-3-nano-30b-a3b (OpenRouter (nvidia/nemotron-3-nano-30b-a3b)) | 26 (13 tasks x 2) | 0 | 0.031 | 5.5 | 1.04 | 2026-09-07 |
| SalesBench-100 | nemotron-3-nano-30b-a3b (OpenRouter (nvidia/nemotron-3-nano-30b-a3b)) | 22 (11 tasks x 2) | 0 | 0.140 | 20.8 | 4.59 | 2026-09-07 |
| DevOpsBench-100 | nemotron-3-nano-30b-a3b (OpenRouter (nvidia/nemotron-3-nano-30b-a3b)) | 24 (12 tasks x 2) | 0 | 0.111 | 17.6 | 0.75 | 2026-09-07 |

# Licensing

Server code: Apache-2.0, Copyright (c) 2026 Blobfish AI (vendored; see VENDORING.md). Task data:
Creative Commons Attribution 4.0 International.
