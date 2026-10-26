/**
 * A headless check on the console, with no browser and no test framework.
 *
 *   node scripts/check_console.mjs
 *   node scripts/check_console.mjs --report reports/report.json
 *
 * Two kinds of check, and the first one is the reason this exists.
 *
 * **Static.** Every id the script reaches for with getElementById must exist
 * as an id in the markup. A typo there is silent in the browser -- the
 * element is null, the handler throws inside an async function, and the
 * console just stops updating with nothing in the terminal. That class of bug
 * is invisible to the Python tests and to anyone who does not open the page.
 * It also asserts the page loads nothing from the network: no CDN, no font
 * host, no analytics. A review gate that phones out while you read its
 * verdict is not a thing to ship, and offline is where this gets demoed.
 *
 * **Behavioural.** The script's pure helpers are evaluated in a context with
 * no DOM at all and run against a gate report -- a real one from the API when
 * `--report` is given, the canned one below otherwise. They are the part that
 * turns numbers into sentences, which is the part that can be quietly wrong:
 * a p-value printed as 0.000, an unbounded interval drawn as zero, a verdict
 * whose explanation contradicts its badge.
 */

import { readFileSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import vm from "node:vm";

const here = dirname(fileURLToPath(import.meta.url));
const root = resolve(here, "..");
const html = readFileSync(resolve(root, "web/index.html"), "utf8");

let passed = 0;
const failures = [];

function check(name, fn) {
  try {
    fn();
    passed += 1;
  } catch (err) {
    failures.push({ name, message: err.message });
  }
}

function assert(condition, message) {
  if (!condition) throw new Error(message || "assertion failed");
}

function assertEqual(actual, expected, message) {
  if (actual !== expected) {
    throw new Error(
      (message ? message + ": " : "") +
        `expected ${JSON.stringify(expected)}, got ${JSON.stringify(actual)}`
    );
  }
}

function assertIncludes(haystack, needle, message) {
  assert(
    String(haystack).includes(needle),
    (message ? message + ": " : "") +
      `expected to find ${JSON.stringify(needle)} in ${JSON.stringify(
        String(haystack).slice(0, 220)
      )}`
  );
}

// -- the script, evaluated without a DOM ----------------------------------

const scriptMatch = html.match(/<script>([\s\S]*?)<\/script>/);
assert(scriptMatch, "no inline <script> found in web/index.html");
const source = scriptMatch[1];

const sandbox = { module: { exports: {} }, console };
vm.createContext(sandbox);
new vm.Script(source, { filename: "index.html#script" }).runInContext(sandbox);
const V = sandbox.module.exports;
assert(V && typeof V.fmtP === "function", "the helpers did not export");

// -- static checks ---------------------------------------------------------

const declaredIds = new Set(
  [...html.matchAll(/\bid="([^"]+)"/g)].map((m) => m[1])
);
const referencedIds = new Set([
  ...[...source.matchAll(/\$\("([^"]+)"\)/g)].map((m) => m[1]),
  ...[...source.matchAll(/getElementById\("([^"]+)"\)/g)].map((m) => m[1]),
]);

check("every id the script reaches for exists in the markup", () => {
  const missing = [...referencedIds].filter((id) => !declaredIds.has(id));
  assertEqual(missing.join(", "), "", "ids referenced but never declared");
  assert(referencedIds.size > 10, "suspiciously few ids referenced");
});

check("the page loads nothing from the network", () => {
  const external = [...html.matchAll(/(?:src|href)="([^"]+)"/g)]
    .map((m) => m[1])
    .filter((url) => /^(https?:)?\/\//.test(url));
  assertEqual(external.join(", "), "", "external resources found");
  assert(!/<link\b/.test(html), "no stylesheet links expected");
});

check("every API path the console calls is one the server serves", () => {
  const paths = [...source.matchAll(/api\("(\/api[^"?]*)/g)].map((m) => m[1]);
  const served = [
    "/api/presets",
    "/api/gate",
    "/api/history",
    "/api/validation",
  ];
  const unknown = paths.filter((p) => !served.includes(p));
  assertEqual(unknown.join(", "), "", "console calls an endpoint not in the API");
  assert(paths.includes("/api/gate"), "the console never runs the gate");
});

check("the markup declares the sections the renderers unhide", () => {
  for (const id of [
    "statsection",
    "chartsection",
    "metricsection",
    "calsection",
    "validationsection",
  ]) {
    assert(declaredIds.has(id), `missing section #${id}`);
    assert(
      new RegExp(`id="${id}"[^>]*hidden`).test(html),
      `#${id} should start hidden`
    );
  }
});

// -- number formatting -----------------------------------------------------

check("a p-value is never printed as zero", () => {
  assertEqual(V.fmtP(0.0004), "<0.001");
  assertEqual(V.fmtP(0), "<0.001");
  assertEqual(V.fmtP(0.0312), "0.031");
  assertEqual(V.fmtP(0.0051), "0.0051");
  assertEqual(V.fmtP(null), "--");
});

check("an unbounded interval says so rather than reading as zero", () => {
  assertEqual(V.fmtCi([null, null]), "unbounded");
  assertEqual(V.fmtCi(null), "unbounded");
  assertEqual(V.fmtCi([-0.1234, 0.5]), "[-0.123, 0.500]");
});

check("signed numbers keep their sign", () => {
  assertEqual(V.fmtNum(0.5), "+0.500");
  assertEqual(V.fmtNum(-0.5), "-0.500");
  assertEqual(V.fmtNum(null), "--");
  assertEqual(V.fmtPct(0.625), "63%");
});

// -- the report fixture ----------------------------------------------------

function cannedReport(overrides = {}) {
  const steps = [];
  for (let n = 1; n <= 200; n++) {
    const radius = n < 3 ? Infinity : 4 / Math.sqrt(n);
    steps.push({
      n,
      mean: -0.2,
      lower: n < 3 ? null : -0.2 - radius,
      upper: n < 3 ? null : -0.2 + radius,
      decision: n === 200 ? "regression" : "inconclusive",
    });
  }
  return Object.assign(
    {
      run_id: 3,
      verdict: "regression",
      reason: "stopped at case 200 of 600: the always-valid interval is below the margin",
      summary: "REGRESSION on 600 cases",
      n_cases: 600,
      judge: "rubric-judge",
      comparisons: 14,
      flagged: ["rubric score"],
      flagged_uncorrected: ["rubric score", "pass rate"],
      resolvable_margin: 0.31,
      labels_to_certify: 280,
      constant_judge_kappa: 0.0,
      correction: {
        method: "benjamini-hochberg",
        alpha: 0.025,
        rejected: ["rubric score"],
        naive_rejected: ["rubric score", "pass rate"],
        family_error_if_uncorrected: 0.2984,
      },
      calibration: {
        n: 286,
        kappa: 0.42,
        kappa_ci: [0.36, 0.48],
        weighted_kappa: 0.652,
        weighted_kappa_ci: [0.593, 0.71],
        exact_agreement: 0.55,
        within_one: 0.97,
        bias: 0.26,
        bias_ci: [0.19, 0.33],
        confusion: [
          [1, 2, 0, 0, 0],
          [0, 8, 6, 1, 0],
          [0, 3, 30, 14, 1],
          [0, 0, 12, 60, 22],
          [0, 0, 1, 18, 107],
        ],
        ceiling: 0.736,
        relative: 0.886,
        required: 0.552,
        threshold: 0.75,
        floor: 0.4,
        trustworthy: true,
        verdict: "weighted kappa 0.65: good enough to rule",
        note: "",
      },
      metrics: [
        {
          label: "rubric score",
          kind: "ordinal",
          higher_is_better: true,
          baseline: 4.12,
          candidate: 3.61,
          delta: -0.51,
          worse: true,
          n: 600,
          test: {
            name: "wilcoxon+bootstrap",
            statistic: 1200,
            p_value: 0.0000004,
            effect: -0.51,
            ci: [-0.62, -0.4],
            informative: 388,
            detail: "normal approximation, 388 non-tied pairs",
          },
          p_adjusted: 0.0000056,
          threshold: 0.0018,
          rejected: true,
          flagged: true,
          flagged_uncorrected: true,
        },
        {
          label: "pass rate",
          kind: "binary",
          higher_is_better: true,
          baseline: 0.71,
          candidate: 0.66,
          delta: -0.05,
          worse: true,
          n: 600,
          test: {
            name: "mcnemar",
            statistic: 9.1,
            p_value: 0.03,
            effect: -0.05,
            ci: [-0.09, -0.01],
            informative: 120,
            detail: "continuity-corrected chi-square on 120 discordant pairs",
          },
          p_adjusted: 0.21,
          threshold: 0.0036,
          rejected: false,
          flagged: false,
          flagged_uncorrected: true,
        },
      ],
      sequential: {
        decision: "regression",
        n_used: 200,
        n_available: 600,
        effect: -0.2,
        ci: [-0.5, -0.06],
        tolerance: 0.05,
        alpha: 0.025,
        stopped_early: true,
        saved: 400,
        saved_fraction: 0.667,
        steps,
      },
      sequential_peeking: {
        decision: "regression",
        n_used: 24,
        n_available: 600,
        effect: -0.3,
        ci: [-0.55, -0.07],
        tolerance: 0.05,
        alpha: 0.05,
        stopped_early: true,
        saved: 576,
        saved_fraction: 0.96,
        steps: steps.slice(1, 40),
      },
      sequential_fixed: {
        decision: "regression",
        n_used: 600,
        n_available: 600,
        effect: -0.507,
        ci: [-0.62, -0.39],
        tolerance: 0.05,
        alpha: 0.025,
        stopped_early: false,
        saved: 0,
        saved_fraction: 0,
        steps: [],
      },
      sequential_subgaussian: {
        decision: "inconclusive",
        n_used: 600,
        n_available: 600,
        effect: -0.2,
        ci: [-0.9, 0.5],
        tolerance: 0.05,
        alpha: 0.025,
        stopped_early: false,
        saved: 0,
        saved_fraction: 0,
        steps,
      },
      config: {
        alpha: 0.05,
        tolerance: 0.05,
        label_fraction: 0.5,
        kappa_threshold: 0.75,
        correction: "bh",
        slice_by_tag: true,
        min_slice: 25,
        min_n: 20,
      },
    },
    overrides
  );
}

const reportFlag = process.argv.indexOf("--report");
const report =
  reportFlag > -1
    ? JSON.parse(readFileSync(process.argv[reportFlag + 1], "utf8"))
    : cannedReport();
const live = reportFlag > -1;

// -- the verdict banner ----------------------------------------------------

check("the badge and its explanation agree for every verdict", () => {
  for (const verdict of ["pass", "regression", "inconclusive", "abstain"]) {
    const copy = V.verdictCopy(
      cannedReport({ verdict, reason: "because " + verdict })
    );
    assertEqual(copy.tone, verdict);
    assertEqual(copy.title, verdict);
    assert(copy.lead.length > 20, `${verdict} has no explanation`);
    assertIncludes(copy.reason, verdict);
  }
});

check("an inconclusive verdict says how fine a margin was reachable", () => {
  const copy = V.verdictCopy(cannedReport({ verdict: "inconclusive" }));
  assertIncludes(copy.lead, "0.310");
  assertIncludes(copy.lead, "and no finer");
});

check("an abstain explains itself with the label count", () => {
  const copy = V.verdictCopy(cannedReport({ verdict: "abstain" }));
  assertIncludes(copy.lead, "286");
  assertIncludes(copy.lead, "280");
  assertIncludes(copy.lead, "human labels");
});

// -- the stat strip and the metric table -----------------------------------

check("the stat strip reports the saving and both flag counts", () => {
  const tiles = V.statTiles(report);
  const joined = tiles.map((t) => t.label + "=" + t.value).join("|");
  assert(tiles.length >= 5, "too few tiles");
  assertIncludes(joined, "cases judged");
  assertIncludes(joined, "uncorrected");
  assertIncludes(joined, "false alarm if uncorrected");
});

check("a comparison lost to the correction is marked, not dropped", () => {
  const rows = V.metricRows(report);
  assertEqual(rows.length, report.metrics.length, "a comparison went missing");

  // The invariant, which holds for a clean run as much as a flagged one: the
  // table's flagged rows are exactly the report's, and the rows marked
  // "uncorrected only" are exactly the difference between the two sets.
  const flagged = rows.filter((r) => r.flagged).map((r) => r.label);
  assertEqual(flagged.join("|"), report.flagged.join("|"), "flagged rows differ");
  const lost = rows.filter((r) => r.lostToCorrection).map((r) => r.label);
  const expectedLost = report.flagged_uncorrected.filter(
    (label) => !report.flagged.includes(label)
  );
  assertEqual(lost.sort().join("|"), expectedLost.sort().join("|"), "near-misses differ");

  for (const row of rows) {
    assert(row.direction === "worse" || row.direction === "better");
    assert(Number.isFinite(row.baseline) && Number.isFinite(row.candidate));
  }
});

check("the metric note names what the correction removed", () => {
  const note = V.metricNote(report);
  assertIncludes(note, "alpha/2");
  assertIncludes(note, String(report.comparisons));
  if (!live) assertIncludes(note, "pass rate");
});

check("the metric note is honest when nothing was removed", () => {
  const clean = cannedReport({
    flagged: ["rubric score"],
    flagged_uncorrected: ["rubric score"],
  });
  assertIncludes(V.metricNote(clean), "No comparison was lost");
});

// -- the chart -------------------------------------------------------------

check("the chart's domain survives unbounded early steps", () => {
  const curves = V.chartCurves(report).map((c) => c.data);
  const domain = V.domain(curves, report.config.tolerance);
  assert(Number.isFinite(domain.lo) && Number.isFinite(domain.hi), "domain is not finite");
  assert(domain.lo < domain.hi, "domain is inverted");
  assert(domain.lo >= -2.5 && domain.hi <= 2.5, "domain was not clamped");
  assert(domain.lo < -report.config.tolerance, "the margin is off the chart");
});

check("a path skips the steps with no interval instead of drawing zero", () => {
  const domain = V.domain([report.sequential], 0.05);
  const scales = V.scales(domain, 600, 980, 320, { l: 54, r: 14, t: 14, b: 34 });
  const steps = [
    { n: 1, mean: 0, lower: null, upper: null },
    { n: 2, mean: 0, lower: Infinity, upper: Infinity },
    { n: 3, mean: -0.1, lower: -0.5, upper: 0.3 },
    { n: 4, mean: -0.1, lower: -0.4, upper: 0.2 },
  ];
  const d = V.path(steps, "lower", scales);
  assertEqual((d.match(/[ML]/g) || []).length, 2, "nulls should not be drawn");
  assert(d.startsWith("M"), "a path must start with a moveto");
  assert(!/NaN|Infinity/.test(d), "the path contains a non-number");
});

check("a band is empty when no step has two finite bounds", () => {
  const scales = V.scales({ lo: -1, hi: 1 }, 10, 100, 100, { l: 0, r: 0, t: 0, b: 0 });
  assertEqual(V.band([{ n: 1, lower: null, upper: null }], scales), "");
  const band = V.band([
    { n: 1, lower: -0.2, upper: 0.2 },
    { n: 2, lower: -0.1, upper: 0.1 },
  ], scales);
  assert(band.endsWith("Z"), "a band must be a closed path");
  assert(!/NaN/.test(band), "the band contains a non-number");
});

check("the chart legend covers the valid rule, the invalid one and Hoeffding", () => {
  const curves = V.chartCurves(report);
  const names = curves.map((c) => c.name).join("|");
  assertIncludes(names, "empirical-Bernstein");
  assertIncludes(names, "invalid");
  assertIncludes(names, "sigma=4");
  assertEqual(new Set(curves.map((c) => c.colour)).size, curves.length);
});

check("the chart note quantifies what peeking costs when peeking resolved", () => {
  const note = V.chartNote(report);
  assertIncludes(note, "eeking", "the note never mentions the invalid rule");
  if (report.sequential_peeking.decision !== "inconclusive") {
    // Stopping early on the invalid rule is only interesting next to its
    // false-alarm rate, so the number has to travel with the claim.
    assertIncludes(note, "46.7%");
    assertIncludes(note, String(report.sequential_peeking.n_used));
  }
  if (report.sequential.decision !== "inconclusive") {
    assertIncludes(note, String(report.sequential.n_used));
  } else {
    assertIncludes(note, "never cleared the margin");
  }
});

check("the chart note is right for a run where nothing resolved", () => {
  const quiet = cannedReport({
    sequential: Object.assign({}, report.sequential, {
      decision: "inconclusive",
      n_used: 600,
      stopped_early: false,
    }),
    sequential_peeking: Object.assign({}, report.sequential_peeking, {
      decision: "inconclusive",
      n_used: 600,
    }),
  });
  const note = V.chartNote(quiet);
  assertIncludes(note, "never cleared the margin");
  assertIncludes(note, "Peeking did not resolve");
  assert(!note.includes("46.7%"), "the peeking rate belongs to a peeking stop");
});

// -- calibration -----------------------------------------------------------

check("the kappa bar puts the judge, its bar and the ceiling in order", () => {
  const geo = V.kappaGeometry(report.calibration);
  assert(geo.lo <= geo.point && geo.point <= geo.hi, "the point is outside its interval");
  assert(geo.ceiling > geo.point, "the ceiling should be above the judge");
  assert(geo.required < geo.point, "this fixture's judge clears its bar");
  for (const value of [geo.lo, geo.hi, geo.point, geo.required, geo.ceiling]) {
    assert(value >= 0 && value <= 100, "a bar position is off the track");
  }
});

check("a judge below its bar puts the marker above the estimate", () => {
  const geo = V.kappaGeometry({
    weighted_kappa: 0.5,
    weighted_kappa_ci: [0.4, 0.6],
    required: 0.59,
    ceiling: 0.79,
  });
  assert(geo.required > geo.point, "the bar should sit above an uncertified judge");
});

check("the calibration panel shows the constant judge next to the real one", () => {
  const lines = V.calibrationLines(report).map((l) => l.label).join("|");
  assertIncludes(lines, "always says 4");
  assertIncludes(lines, "inter-annotator ceiling");
  assertIncludes(lines, "share of the ceiling");
  assertIncludes(lines, "bias");
});

check("the calibration verdict matches whether the judge was certified", () => {
  const certified = V.calibrationVerdict(report);
  assertIncludes(certified, report.calibration.trustworthy ? "Certified" : "Not certified");
  const refused = V.calibrationVerdict(
    cannedReport({
      calibration: Object.assign({}, report.calibration, {
        trustworthy: false,
        weighted_kappa_ci: [0.45, 0.62],
        required: 0.59,
      }),
    })
  );
  assertIncludes(refused, "Not certified");
  assertIncludes(refused, "not enough labels");
});

// -- history and the validation panel --------------------------------------

check("history rows are formatted for reading, not for parsing", () => {
  const rows = V.historyRows([
    {
      run_id: 7,
      created_ms: 1761500000000,
      label: "release-7",
      suite: "synthetic",
      verdict: "regression",
      effect: -0.507,
      ci: [-0.62, -0.39],
      kappa: 0.652,
      judge: "rubric-judge",
      flagged: 2,
      n_cases: 600,
      n_used: 225,
      stopped_early: true,
    },
  ]);
  assertEqual(rows.length, 1);
  assertEqual(rows[0].effect, "-0.507");
  assertEqual(rows[0].cases, "225/600");
  assertEqual(rows[0].kappa, "0.65");
  assertEqual(rows[0].judge, "rubric-judge");
  assert(rows[0].when.length > 0, "a run with no readable time");
});

check("the validation panel degrades to nothing without a study", () => {
  assertEqual(V.validationLines(null).length, 0);
  assertEqual(V.validationLines({}).length, 0);
});

check("the validation panel reports the measured rates", () => {
  const lines = V.validationLines({
    quick: false,
    peeking: {
      "fixed horizon (one look)": { error_rate: 0.059, mean_n: 600 },
      "peeking after every case": { error_rate: 0.467, mean_n: 366 },
    },
    multiplicity: {
      comparisons: 14,
      uncorrected_alpha: 0.51,
      benjamini_hochberg: 0.027,
    },
    pairing: { paired_power: 0.983, unpaired_power: 0.003 },
    calibration: { certified_from: 280 },
  });
  const joined = lines.map((l) => l.label + "=" + l.value).join("|");
  assertIncludes(joined, "47%");
  assertIncludes(joined, "6%");
  assertIncludes(joined, "51%");
  assertIncludes(joined, "280");
});

// -- report -----------------------------------------------------------------

const label = live ? `live report (${process.argv[reportFlag + 1]})` : "canned report";
if (failures.length) {
  console.error(`console check FAILED on the ${label}`);
  for (const failure of failures) {
    console.error(`  x ${failure.name}\n      ${failure.message}`);
  }
  console.error(`\n${passed} passed, ${failures.length} failed`);
  process.exit(1);
}
console.log(`console check passed on the ${label}: ${passed} checks`);
