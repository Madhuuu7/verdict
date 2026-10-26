# verdict

A regression gate for LLM pipelines. It answers one question -- *is this
candidate worse than what we ship today?* -- and it is built around the four
ways the usual answer is wrong.

| the usual eval gate | what this does instead |
|---|---|
| trusts the judge | calibrates it against human labels first, **relative to the inter-annotator ceiling**, and abstains when the judge is not good enough to rule |
| compares two means | **pairs on the case**, so shared case difficulty stops being noise |
| tests fifteen things at 0.05 | controls the **false discovery rate** across the whole family |
| peeks at the p-value | uses a **confidence sequence**, valid at every sample size, so stopping early is allowed |

Every error rate below is **measured**, not cited. `scripts/validate.py` runs
the simulations that produce them and writes the numbers the console displays.

---

## The numbers

From `python scripts/validate.py` (30 seconds, alpha = 0.05). Reproduce them
and they should match to Monte-Carlo noise.

### Stopping early is only free if the interval was built for it

1,500 null experiments, 600 cases each, no real difference between the
variants. Any definite verdict is a mistake.

| stopping rule | false alarms | cases read |
|---|---|---|
| one look, at the end | **5.9%** | 600 |
| re-test after every case | **46.7%** | 366 |
| sub-Gaussian confidence sequence (sigma=4) | **0.0%** | 600 |
| empirical-Bernstein confidence sequence | **0.0%** | 600 |

Re-testing after every case is the procedure almost everyone writes. It looks
identical in a dashboard and it is wrong nine times out of ten that it fires.

### What that validity costs

1,000 experiments with a real -0.25 effect:

| stopping rule | detected | cases read | saved |
|---|---|---|---|
| one look, at the end | 100% | 600 | 0% |
| sub-Gaussian sequence (sigma=4) | **0.0%** | 600 | 0% |
| empirical-Bernstein sequence | **96.1%** | 403 | **32.8%** |

The sub-Gaussian row is why this project has two sequence implementations. A
rubric difference is bounded by 4, so the only *honest* Hoeffding proxy is
sigma = 4 -- and that boundary is correct and useless: it reports
`inconclusive` for a regression a fixed-horizon test calls obvious. The
empirical-Bernstein boundary gets to use the variance it observes, and pays
for it with a transcribed constant whose coverage is therefore measured rather
than trusted. (On the bundled suite the gate settles a clear quality drop at
**case 225 of 600 -- 62% of the model calls never made**.)

### Pairing

300 experiments, 120 cases, case difficulty sd 1.0 shared by both variants,
per-case noise sd 0.3, true effect -0.15:

| | detected | mean interval width |
|---|---|---|
| paired bootstrap | **98.3%** | 0.150 |
| unpaired bootstrap | **0.3%** | 0.519 |

Same data, same point estimate, 3.5x the interval. An unpaired comparison has
to treat case difficulty as noise; a paired one differences it away.

### Testing fourteen things at once

4,000 families of 14 all-null comparisons -- the number the gate makes on the
bundled suite with slicing on:

| | at least one false alarm |
|---|---|
| uncorrected, alpha = 0.05 | **51.0%** (formula: 51.2%) |
| uncorrected, alpha/2 | 29.8% |
| Benjamini-Hochberg at alpha/2 | **2.7%** |
| Holm at alpha/2 | 2.7% |

A gate that cries wolf on half of all clean runs gets switched off, and then
the absence of an alarm means nothing either.

### How many human labels the judge needs

The judge is scored against annotator A; the ceiling is A against B on the
same cases.

| labels | ceiling | judge | lower bound | bar | share of ceiling | certified |
|---|---|---|---|---|---|---|
| 80 | 0.763 | 0.582 | 0.468 | 0.572 | 76% | no |
| 160 | 0.778 | 0.628 | 0.545 | 0.583 | 81% | no |
| **280** | 0.764 | 0.648 | **0.584** | 0.573 | 85% | **yes** |
| 400 | 0.740 | 0.636 | 0.582 | 0.555 | 86% | yes |
| 600 | 0.736 | 0.654 | 0.610 | 0.552 | 89% | yes |

The judge is close to a second annotator from 80 labels on. What takes 280 is
*proving* it: certification tests the interval's **lower bound**, and below
that the bootstrap interval is wider than the margin between the judge and its
bar. A judge that answers "4" to everything scores kappa **0.000** while
agreeing with the humans more than half the time, which is what kappa is for.

### Each regression shape is caught by a different comparison

600 cases, default config:

| preset | verdict | caught by |
|---|---|---|
| identical pipelines | inconclusive | *nothing*, across 14 comparisons |
| one slice broken | regression | `rubric score [math]` **and nothing else** |
| slower, same quality | regression | `mean latency` alone |
| more refusals | regression | `refusal rate`, and the rubric with it |
| worse on hard cases only | regression | the suite rubric; the median barely moves |
| clear quality drop | regression | 12 of 14, stopping at case 225 |

This is the argument for the metric set and for slicing: drop the slices and
the broken-slice row goes unflagged.

---

## Running it

```bash
python -m venv .venv && .venv/Scripts/activate   # or source .venv/bin/activate
pip install -e ".[dev]"

pytest                                   # 261 tests, ~40s
python scripts/validate.py --quick       # the study, a tenth of the trials
node scripts/check_console.mjs           # the console, headless
uvicorn verdict.api.app:app --reload     # the console at http://127.0.0.1:8000
```

The console runs the gate, draws the three stopping rules on the same data,
shows every comparison with its raw and adjusted p-value, and shows the judge
against its bar and the annotator ceiling.

### Against a real model

`SimulatedPipeline` and `RubricJudge` are the defaults. `HttpPipeline` and
`LlmJudge` speak the OpenAI chat format, so they work against OpenAI, Groq,
OpenRouter, Together or a local llama.cpp server:

```bash
export VERDICT_BASE_URL=https://api.groq.com/openai/v1
export VERDICT_API_KEY=...
export VERDICT_MODEL=llama-3.3-70b-versatile
```

Point them at a suite in `suites/example.jsonl`'s format. **This path has
never been run against a paid endpoint** -- there is no key here -- so it is
in the limits below rather than claimed as tested.

---

## Why the pipelines are simulated

Because an error rate can only be measured against a known truth. To say "this
gate flags a regression that is not there 2.7% of the time", you need runs
where you know there is no regression, and a real model cannot give you that.
A simulated pipeline with a dialled-in quality offset can.

The statistical machinery is identical either way -- it only ever sees
numbers. What is simulated is the thing being measured, not the measuring.

---

## Decisions worth arguing about

**The judge's bar is relative to the inter-annotator ceiling, not absolute.**
An absolute bar -- "kappa must clear 0.6" -- was the first design and it was
wrong. Human labels carry their own noise, and no judge can agree with a noisy
reference more than a second annotator does. Two simulated annotators reach
weighted kappa ~0.74 here; against an absolute 0.6 bar, driving the judge's
noise to nearly zero still left it failing, because the ceiling was never the
judge. The bar is 0.75 of the measured ceiling, with an absolute floor so that
two annotators who agree on nothing cannot certify a judge by having a low
ceiling.

**Certification tests the lower confidence bound.** With a few dozen labels a
point estimate cannot tell 0.58 from 0.68. This is the whole reason 280 labels
are needed when 80 already show a good judge.

**The error budget is split alpha/2 + alpha/2.** The gate can be wrong twice:
by stopping early, or at the end of the suite. Spending the full alpha on each
would allow 2 alpha overall, so each look gets half. It is a Bonferroni split
across two looks -- crude, and honest about there being two. The combined null
error rate is measured, not assumed.

**Benjamini-Hochberg, not Holm, by default.** When the gate flags three
metrics the useful guarantee is "most of these are real", not "there is a 95%
chance none is spurious". Holm is one config flag away and its rejections are
always a subset.

**Non-inferiority, not a point null.** A gate asking "are these exactly equal"
answers no at large enough n, every time, for any real pipeline. The question
is whether the candidate is worse by more than a margin you declared in
advance.

**`inconclusive` is a real outcome, and it reports a `resolvable_margin`.** A
gate forced to answer yes or no on 40 cases will answer wrongly. When the
suite does not resolve the question, the report says how fine a margin it
*could* have certified -- so "inconclusive" means "add cases or quieten the
judge", not a shrug.

**`abstain` is a real outcome too.** If the judge cannot be certified, nothing
is tested and the gate says so. A number from an instrument nobody checked is
worse than no number, because it will be acted on.

**No SciPy.** The six distribution functions it would have supplied are
written out, each naming the identity it uses, so no p-value in a gate that
blocks releases is opaque. They are tested against closed forms and against
SciPy when it happens to be installed.

---

## Bugs worth keeping in the history

Each of these made the *simulation* subtly wrong rather than obviously broken,
which is the kind that survives review. Each now has a test that measures the
thing that gave it away.

**A salt collision between the refusal draw and the quality noise.** The
refusal used the same salt as the first uniform of the Box-Muller pair, so a
refusal fired exactly when the quality noise was extreme. It barely moved the
mean -- which is why it survived -- but it inflated the refused group's spread
by 45%, and it produced a **false REGRESSION on a pair of identical
pipelines**.

**Two annotators who were the same person.** `HumanLabeller` seeded itself on
the literal `"human"` instead of its own name, so the inter-annotator ceiling
came out at exactly 1.000 -- and every judge was then measured against an
oracle.

**A confidence sequence four times too wide to be useful.** Not a bug, a
discovery: the honest sub-Gaussian proxy for a bounded rubric difference makes
the gate useless, and measuring that is what motivated the
empirical-Bernstein boundary.

**A step curve indexed by position instead of by n.** `peeking` records no
step for n = 1 -- a single observation has no sample standard deviation -- so
on one of four curves the API's thinner ran off the end of the array.

**A `judge` column that always said "judge".** It was filled from the
calibration's note field, which is empty on every successful run, so the one
thing it was for -- telling a judge swap apart from a pipeline regression --
could not be done.

---

## What this does not do

- **Never run against a paid endpoint.** `HttpPipeline` and `LlmJudge` are
  written and unit-tested against a stubbed transport; they have not met a
  real model.
- **The gate runs synchronously.** About a second for 600 simulated cases,
  which the threadpool handles. Against a real model a run is minutes of
  rate-limited calls and wants a job queue.
- **Bootstrap intervals are percentile, not BCa.** Adequate for a mean
  difference, which is close to symmetric; a skewed statistic would want BCa.
- **BH assumes independence or positive dependence.** True for metrics from a
  shared suite, which move together. Arbitrary dependence needs the
  Benjamini-Yekutieli variant and its log(m) penalty.
- **The family-wise correction is applied to fixed-horizon p-values** while
  the stopping decision uses the always-valid interval. The two mechanisms
  compose through the alpha/2 split rather than through a single theory.
- **SQLite, single process.** Fine for a history; not a multi-writer store.
- **The judge's own precision barely moves its measured agreement** on this
  simulation, because grade rounding dominates it. The disagreement you see is
  mostly the human's noise and the judge's bias.

---

## Layout

```
src/verdict/
  stats/distributions.py   normal CDF/PPF, chi-square(1), exact binomial
  stats/paired.py          McNemar, Wilcoxon signed-rank, paired bootstrap
  stats/multiplicity.py    Benjamini-Hochberg, Holm
  stats/sequential.py      confidence sequences, peeking, fixed horizon
  cases.py                 suites, tags, the synthetic generator
  pipelines.py             the thing under test, and five regression shapes
  judges.py                rubric judge, LLM judge, human labellers
  calibration.py           Cohen's kappa, weighted, with bootstrap intervals
  gate.py                  run, judge, calibrate, test, correct, decide
  store.py                 run history in SQLite
  api/app.py               the HTTP surface
web/index.html             the console, self-contained
scripts/validate.py        the simulation study behind every number above
scripts/check_console.mjs  the console's headless check
suites/example.jsonl       the suite format, annotated
```

## License

MIT.
