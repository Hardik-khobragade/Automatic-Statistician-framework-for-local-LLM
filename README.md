# Automatic Statistician v2.0 (LLM-based re-implementation)

A local-first pipeline that takes a dataset + a plain-English task, and
produces a Word report with statistics, charts, and assumption checks --
driven by a small LLM (e.g. **Qwen3-4B**) running entirely on your machine.

```
LAYER 1  data_profiling.py    load + type-detect + descriptive profile -> JSON
LAYER 2  sandbox.py            sandboxed, stateful code execution (the "tool")
         react_agent.py        Thought -> Action -> Observation loop (the "agent")
LAYER 3  stats_toolkit.py      tested scipy/statsmodels/pingouin wrappers
LAYER 4  report_generator.py   assembles the .docx (charts, tables, narrative)
LAYER 5  validation.py         result sanity checks + optional ground-truth diff
main.py                        wires all five layers together (CLI)
```

## Why it's built this way (given a 4B model on a 4GB GPU)

A 4B local model is good at *calling a well-documented function correctly*
and *writing a short paragraph*, and much less reliable at *writing correct
statistics code from scratch* or *holding a long plan in its head*. So:

- The agent is steered to call `stats_toolkit.*` functions (which already
  run the right assumption checks and effect sizes) rather than writing
  raw `scipy`/`statsmodels` code itself.
- `query_data` answers (columns, dtypes, samples) are computed directly in
  Python, not via LLM-authored code -- one less way for a small model to fail.
- Every number that ends up in the final report's tables comes from
  `record_result(...)`, i.e. from the toolkit's own return values -- never
  retyped by the model -- so the report's numbers are correct even if the
  model's prose is mediocre.
- The ReAct loop has loop-detection and a max-iteration cap, since small
  models sometimes get stuck repeating an action.
- A baseline "Data Overview" page (missingness chart, histograms,
  correlation heatmap) is generated deterministically, independent of the
  agent, so the report looks complete even on a so-so run.

## 1. Install

```bash
python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## 2. Set up the local model (Ollama)

[Ollama](https://ollama.com) is the easiest way to serve a quantized model
locally and is what this project assumes by default (it exposes an
OpenAI-compatible API at `http://localhost:11434/v1`). LM Studio and
`llama.cpp`'s `server` also work -- see "Using a different server" below.

```bash
ollama pull qwen3:4b
```

**4GB VRAM tuning.** A 4B model at Ollama's default `q4_K_M` quantization
needs roughly 3-3.5GB for weights, leaving little headroom for the KV cache
on a 4GB card. If you hit out-of-memory errors or it's spilling to CPU
(very slow):

- Keep the context window modest -- this project defaults to 4096 tokens
  (`AUTOSTAT_LLM_CONTEXT_WINDOW`). Match it on the Ollama side:
  ```bash
  OLLAMA_NUM_CTX=4096 ollama serve
  ```
  or set `num_ctx` in a custom Modelfile.
  Make sure to close apps that can hold VRAM (browsers with hardware acceleration, etc).
- Close other GPU applications (browser hardware acceleration, games, etc).
- If it's still too tight, try a smaller quant (`qwen3:4b-q4_0` or similar
  tags on the Ollama library page) or step down to `qwen3:1.7b`.

Verify the model is reachable:

```bash
python setup_check.py
```

## 3. Try it offline first (no GPU needed)

This runs the real sandbox, real stats toolkit, and real report generator
against a scripted fake "LLM" so you can confirm your installation works
before waiting on real model generations:

```bash
python test_pipeline.py
```

It writes `_test_run/test_report.docx` -- open it to check formatting on
your system (Word / LibreOffice / Google Docs all render python-docx output
fine).

## 4. Run a real analysis

```bash
python main.py --data yourfile.csv \
    --task "Find out what predicts customer churn" \
    --output report.docx
```

Useful flags:

| Flag | Default | Purpose |
|---|---|---|
| `--model` | `qwen3:4b` | model name as your server knows it |
| `--base-url` | `http://localhost:11434/v1` | OpenAI-compatible endpoint |
| `--max-iterations` | 20 | ReAct loop step cap |
| `--sandbox-timeout` | 60 | seconds before a single execute_python call is killed |
| `--ground-truth` | - | JSON file of expected values for a benchmark dataset (see below) |
| `--verbose` | off | print every Thought/Action/Observation as it happens |

Outputs land next to `--output`:
- `report.docx` -- the deliverable
- `figures/` -- every chart (agent-made and auto-generated)
- `transcript.md` -- the full ReAct trace, useful for debugging a bad run
- `data_profile.json` -- the Layer-1 profile

### Using a different server

Anything exposing an OpenAI-compatible `/v1/chat/completions` works:

```bash
# LM Studio
python main.py --data x.csv --task "..." --base-url http://localhost:1234/v1 --model <name-in-lm-studio>

# llama.cpp server
python main.py --data x.csv --task "..." --base-url http://localhost:8080/v1 --model <whatever>
```

### Ground-truth comparison (benchmark datasets)

If you're testing against a dataset where you already know the right
answer (simulated data, a textbook example, a previous trusted analysis),
pass a JSON file mapping the `name` you gave `record_result(...)` to
expected field values:

```json
{
  "Income by group": {"p_value": 0.30, "eta_squared": 0.008}
}
```

```bash
python main.py --data x.csv --task "..." --ground-truth expected.json
```

The report gets a "Comparison with Ground Truth" section showing observed
vs. expected and whether each is within a 5% tolerance.

## Troubleshooting: "the report has no results / no findings"

By far the most common cause: **Qwen3 (and other recent models) default to
a hybrid "thinking" mode** that wraps reasoning in `<think>...</think>`
before the real answer. If that goes unhandled, every ReAct step fails to
parse, the agent records nothing, and you get an empty report. This project
already defends against it in three layers:

1. `main.py` passes `think: false` to the server's chat-completions call
   (Ollama's native toggle for this).
2. The system prompt also explicitly asks the model not to use `<think>`.
3. `<think>...</think>` blocks are stripped out of every reply before
   parsing, regardless of whether (1) or (2) worked.

If you still get an empty report:

- Run `python setup_check.py` -- it sends one test prompt through the exact
  same parsing path as a real run and tells you directly whether thinking
  leaked through, whether a reply got cut off mid-`<think>` (raise
  `AUTOSTAT_LLM_MAX_TOKENS`), or whether the model isn't following the
  format at all.
- Check `transcript.md` from the failed run -- it has the raw model output
  for every step, including any thinking content.
- A run that aborts after a handful of steps with an explicit
  `aborted_reason` printed to the console (rather than running all the way
  to `--max-iterations`) means the model never produced a single parseable
  action -- raising `--max-iterations` will not help; the format itself is
  the problem.
- Make sure your Ollama version is recent enough to support the `think`
  request parameter; older versions silently ignore it -- and unlike the
  `<think>...</think>` text-tag case, that's *not* harmless: if `think:false`
  is ignored, the model thinks at full length on every turn, and if it
  burns through the entire `max_tokens` budget before ever producing a real
  answer, `message.content` comes back **completely empty** (no `<think>`
  text to even strip -- there's nothing there). You'll see this directly:
  `main.py`/`setup_check.py` now print a line like
  `[llm_client] empty content returned -- finish_reason=length,
  completion_tokens=1536/1536, think_param_sent=True, ...` whenever this
  happens, which tells you the budget was fully consumed by reasoning with
  zero tokens left for the actual answer. If you see that:
  1. Run `ollama --version` and update Ollama (`winget upgrade Ollama.Ollama`
     on Windows, or redownload from ollama.com) -- `think` support requires
     a fairly recent release.
  2. Raise `AUTOSTAT_LLM_MAX_TOKENS` (env var) well above the default 1536
     as a stopgap -- this costs more time per step on a 4GB GPU but at least
     gives the model room to finish thinking and still answer.
  3. If neither helps, the most reliable fix on constrained hardware is
     switching away from a forced-reasoning model entirely -- try
     `qwen2.5:3b-instruct` or `llama3.2:3b` instead of `qwen3:4b`. Plain
     instruct models (no built-in chain-of-thought) tend to follow a strict
     ReAct template far more reliably *and* run faster, since there's no
     reasoning overhead to suppress in the first place.

## Classification support

As of v2.1.0, Layer 1 tries to guess a likely prediction target in the data
profile (column named `target`/`label`/`class`/etc., or a low-cardinality
categorical column, with a bonus if it's the last column -- the convention
used by sklearn's toy datasets like `wine.csv`) and surfaces it directly in
the prompt: `LIKELY TARGET COLUMN: 'target' (3 classes, confidence=high) --
use multinomial_logistic_regression and/or classification_test`. This saves
the agent several turns of reasoning about which column/test to use on
ambiguous tasks like "do a classification test."

Three classification-relevant toolkit functions, each suited to a different
question:

| Function | Question it answers | Output |
|---|---|---|
| `logistic_regression(df, formula)` | Which predictors matter, for a **binary** outcome? | p-values, odds ratios (rejects >2-class targets with a clear error pointing elsewhere) |
| `multinomial_logistic_regression(df, formula)` | Which predictors matter, for a **3+ class** outcome? | p-values/odds ratios per class vs. a reference class |
| `classification_test(df, target, features=None, model="random_forest")` | How accurately can this be predicted at all? | held-out accuracy, macro precision/recall/f1, confusion matrix, feature importances. Works for binary or multi-class. `model` can also be `"logistic"` or `"decision_tree"`. |

`multinomial_logistic_regression`'s accuracy/confusion matrix are
training-set fit (a model-fit measure), not a held-out predictive estimate
-- `classification_test` does the actual train/test split for that, and
Layer 5 validation flags this distinction automatically in the report.

## Extending the toolkit

Add a new wrapper function to `autostat/stats_toolkit.py` (return a plain
dict, cast numpy scalars with the existing `_round`/`_native` helpers), then
add one line to the cheat-sheet in `autostat/react_agent.py`
(`TOOLKIT_CHEAT_SHEET`) so the agent knows it exists. `report_generator.py`'s
`render_result_dict()` is generic -- it'll render any new dict shape into
tables automatically (scalars -> one table, nested flat dicts -> sub-tables,
dict-of-dicts -> a matrix table, list-of-dicts -> a row-per-item table).

## Honest limitations

- **Sandbox is best-effort, not a hardened security boundary.** It blocks
  obvious things (filesystem/network/process access, `eval`/`exec`) via an
  AST whitelist and runs each step in its own subprocess with a timeout.
  That's enough to contain an accidental bug from a model that isn't trying
  to escape, on your own machine, for your own data. It is *not* sufficient
  isolation for genuinely untrusted/adversarial code -- for that, run the
  whole thing inside a container/VM with no filesystem or network access.
- **A 4B model will sometimes go in circles or under-explore.** The
  loop-detector and report-section nudges help, but check `transcript.md`
  if a report looks thin -- you may need to give a more specific `--task`,
  raise `--max-iterations`, or just re-run (temperature is low but local
  models still vary run to run).
- **The agent picks the tests; it doesn't always pick the *best* test.**
  The toolkit reports the relevant assumption checks and nonparametric
  fallback alongside every parametric result specifically so you can
  override the agent's choice when reading the report.
- **Treat this as a fast first pass, not a substitute for review** before
  any decision-critical conclusion -- the same caveat you'd apply to any
  analysis automation, LLM-driven or not.