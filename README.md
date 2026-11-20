# Online Experimentation Platform

[![CI](https://github.com/rkesseku/experimentation-platform/actions/workflows/ci.yml/badge.svg)](https://github.com/rkesseku/experimentation-platform/actions/workflows/ci.yml)

A Python library, FastAPI service and Streamlit dashboard for **designing, monitoring and analysing online A/B tests** with statistically defensible defaults: variance-reduced power analysis, deterministic bucketing with sample-ratio-mismatch checks, CUPED/CUPAC and delta-method analysis, anytime-valid sequential monitoring, multiplicity control, guardrail non-inferiority tests, and heterogeneous-effect analysis.

All data in this repository is **simulated**. The simulator has known ground truth, so every statistical claim below is checked by the test suite rather than asserted.

## The problem

Most experimentation mistakes are procedural rather than arithmetic:

* **Peeking.** Teams watch a dashboard and stop as soon as `p < 0.05`. With 50 looks at nominal 5%, the real false-positive rate is about a third (see Results).
* **Wrong variance.** Ratio metrics such as revenue per session, when analysed per row, ignore within-user correlation. Heavy-tailed metrics make the t-test under-powered.
* **Broken randomisation.** A logging bug that drops 3% of the treated users is invisible in the effect estimate and fatal to its validity. A sample-ratio-mismatch (SRM) check catches it for free.
* **Many metrics, one story.** Ten metrics tested at 5% give a 40% chance of a spurious "win".
* **Wasted sample.** Pre-period data can remove a quarter to a half of the metric variance.

This project packages the countermeasures behind one readout.

## Architecture

```
src/expplatform/
  design.py       power, sample size, MDE (means, proportions, ratio metrics, r2-aware)
  assignment.py   SHA-256 bucketing, salts, mutually-exclusive layers, SRM chi-square
  analysis.py     Welch, two-proportion, delta method, CUPED, Lin regression (HC2),
                  CUPAC (cross-fitted GBM covariate), percentile/BCa bootstrap,
                  Holm / Benjamini-Hochberg, non-inferiority
  sequential.py   mSPRT + confidence sequences, Lan-DeMets O'Brien-Fleming bounds,
                  peeking simulation
  cate.py         corrected segment effects (+ Cochran Q), cross-fitted T-learner
                  (scikit-learn or EconML backend) with randomisation-based validation
  simulator.py    ground-truth generator: heavy tails, novelty, HTE, SRM bugs, ratio metrics
  report.py       build_readout(): one JSON for SRM, effects, guardrails, sequential, segments
  api.py          FastAPI service           dashboard.py   Streamlit readout
  config.py       pydantic experiment/metric specs
tests/            55 tests; Monte-Carlo checks use fixed seeds and stated tolerances
examples/reproduce_results.py   regenerates every number in this README
```

```
 events (CSV/parquet) --> ingest --> SRM + assignment audit --> adjusted effects
                                          |                       --> Holm/BH over primary+secondary
                                          |                       --> guardrail NI tests
                                          +--> sequential path (mSPRT CS, OBF checkpoints) --> readout JSON --> dashboard
```

## Methods

**Design.** For a mean difference `delta` with per-unit sd `sigma`, allocation ratio `k = n_t/n_c` and covariate `R^2 = r2`:
`n_c = (z_{1-a/2} + z_{power})^2 * sigma^2 (1 - r2)(1 + 1/k) / delta^2`.
Proportions use the pooled-null / unpooled-alternative formula. MDE solves the exact power function. Ratio metrics are sized from the delta-method per-user variance.

**Assignment.** Each unit gets `u = SHA256(salt:unit)[:8] / 2^64`, and variants are cumulative-weight intervals of `u`. A `Layer` hashes units into 10,000 buckets and gives each experiment a disjoint bucket range (mutual exclusion). Different layer salts give independent experiments. SRM is a chi-square goodness-of-fit test of observed arm counts against the intended split, flagged at `p < 0.001`.

**Analysis.**
* Welch t-test with Satterthwaite degrees of freedom, and a two-proportion z test (pooled p-value, Wald CI).
* Ratio metrics `R = sum N / sum D` per arm: `Var(R) ~ [Var(N) - 2R Cov(N,D) + R^2 Var(D)] / (n * mean(D)^2)` (delta method), arms independent.
* CUPED: `Y_adj = Y - theta (X - mean X)` with `theta = Cov(Y,X)/Var(X)` pooled within arm. Variance shrinks by `rho^2`, and the bias is zero because `X` is pre-treatment. CUPAC replaces `X` with a **cross-fitted** gradient-boosting prediction from pre-period features. Lin's interacted regression with HC2 errors handles several covariates.
* Bootstrap: percentile and BCa (jackknife acceleration) for heavy-tailed metrics.
* Multiplicity: Holm (FWER) or Benjamini-Hochberg (FDR) across primary + secondary metrics. Guardrails use one-sided non-inferiority tests, `H0: delta <= -margin`. An intersection-union "all guardrails pass" claim needs no correction.

**Sequential monitoring.** With estimate `D_n ~ N(theta, V_n)` and a `N(0, tau^2)` mixing prior,
`Lambda_n = sqrt(V_n/(V_n + tau^2)) * exp( tau^2 D_n^2 / (2 V_n (V_n + tau^2)) )`
is a non-negative martingale under H0, so by Ville's inequality `P(sup_n Lambda_n >= 1/alpha) <= alpha`. The test rejects the first time `Lambda_n >= 1/alpha`, with the always-valid p-value `min(1, 1/max_j Lambda_j)`. Inverting gives a confidence sequence with radius `r^2 = 2V(V + tau^2)/tau^2 * (log(1/alpha) + 0.5 log((V + tau^2)/V))`. For a planned number of looks the library also computes Lan-DeMets O'Brien-Fleming-type boundaries (spending `a(t) = 2 - 2 Phi(z_{a/2} / sqrt t)`) by numerical integration of the surviving Brownian-motion sub-density, and the tests check them by Monte Carlo simulation. In the readout, `tau` defaults to 0.1x the metric sd unless pre-specified. It affects power but not validity. The variance is re-estimated at each look (a plug-in, asymptotic version of the mSPRT).

**Heterogeneity.** Per-segment Welch effects are multiplicity-corrected and accompanied by Cochran's Q omnibus test. The CATE option is a cross-fitted T-learner (`mu1(x) - mu0(x)`, scikit-learn or EconML `TLearner`). Units are grouped into out-of-fold CATE quantiles, and the *realised* effect per group is estimated by an ordinary Welch test. That test is valid under randomisation and shows whether the model finds real heterogeneity.

## Quick start

```bash
pip install -e ".[dev,dashboard]"          # add ",econml" for the EconML backend
pytest -q                                   # ~40-60 s
python -m expplatform.demo                  # text readout on simulated data
streamlit run src/expplatform/dashboard.py  # dashboard (toggle an SRM bug in the sidebar)
uvicorn expplatform.api:app --reload        # API docs at http://127.0.0.1:8000/docs
python examples/reproduce_results.py        # regenerate the Results numbers (~75 s)
```

```python
from expplatform import design
from expplatform.sequential import peeking_simulation
from expplatform.simulator import SimConfig, simulate_experiment

design.sample_size_means(delta=0.2, sd=1.0, r2=0.25)      # SampleSize(n_control=295, ...)
df = simulate_experiment(SimConfig(n=40_000, effect=0.25, rho=0.5, srm_drop=0.0, seed=7))
peeking_simulation(effect=0.0, seed=2026)                 # naive vs mSPRT vs O'Brien-Fleming
```

API flow:

```bash
curl -X POST localhost:8000/experiments -H 'content-type: application/json' -d @config.json   # -> {"id": ...}
curl -X POST "localhost:8000/experiments/$ID/events?format=csv" --data-binary @events.csv       # or format=parquet
curl localhost:8000/experiments/$ID/analysis
```

Events are unit-level rows (`unit_id`, `variant`, optional `ts`, metric and covariate columns). Other endpoints: `POST /design/sample-size`, `GET /experiments/{id}/assign/{unit_id}`.

## Results

Produced by `examples/reproduce_results.py` (simulated data, seeds stated, output in `examples/results.json`).

**Peeking vs sequential methods, no true effect** (50 looks, 100 new units per arm per look, sd 1, alpha = 0.05, 20,000 simulations, seed 2026, with the arm-mean difference simulated exactly and the variance known):

| Procedure | False-positive rate |
|---|---|
| Fixed-horizon test, one look at the end | 0.049 |
| **Naive peeking** (|z| > 1.96 at any of 50 looks) | **0.322** |
| **mSPRT** (tau = 0.2), any look | **0.017** |
| O'Brien-Fleming (Lan-DeMets), 5 equally spaced looks | 0.048 |

With the variance **re-estimated at every look** from raw simulated data (30 looks, 1,000 simulations, seed 2028): fixed 0.062, naive peeking 0.297, mSPRT 0.022, O'Brien-Fleming 0.054 (Monte-Carlo sd of a rate near 0.05 is about 0.007 at this size).
Power with a true effect of 0.1 sd (20,000 simulations, seed 2027, mSPRT tau = 0.1): fixed 0.999, mSPRT 0.982, O'Brien-Fleming 0.999 (this effect is large for the sample sizes used). Among rejecting runs, mSPRT stopped on average after 36% of the maximum sample. mSPRT is conservative. It pays for validity under continuous monitoring with a lower rejection rate and a wider interval than a single look. O'Brien-Fleming 5-look boundaries: 4.383, 3.100, 2.553, 2.254, 2.064.

**CUPED / CUPAC** (500 simulated experiments of n = 2,000, true effect 0.4, corr(x_pre, y) = 0.5, seeds 0-499): mean CUPED estimate 0.405 vs 0.401 unadjusted (Monte-Carlo se 0.003), so there is no detectable bias. The variance ratio CUPED / unadjusted is **0.735** (theory `1 - rho^2` = 0.75), a **26.5%** variance reduction. On 8 datasets of n = 8,000 where outcomes depend non-linearly on pre-period features, variance reduction was 25.2% with the single pre-period covariate and **45.6%** with a cross-fitted gradient-boosting CUPAC covariate.

**Calibration.** Welch type-I error under the null was 0.0472 (5,000 sims, n = 200/arm, unequal variances, seed 31). For power, the calculator asks for 566 per arm for delta = 0.5, sd = 3, power 0.8, and simulated power was 0.802 (3,000 sims, seed 32). Delta-method 95% CI coverage for a ratio-of-means effect was 0.9485 (2,000 simulated experiments, 1,500 users/arm, seed 33). For sizing, delta = 0.2 sd needs 393/arm, or 295/arm with `r2 = 0.25`, and detecting a lift from 10% to 12% conversion needs 3,841/arm.

**SRM detection** (n = 40,000, 200 simulations per row, random treated-user loss, threshold p < 0.001). The false-alarm rate was 0.0 with no loss. The check flagged 1% of runs at 1% loss, 8% at 2% loss and 94% at 5% loss. Small losses are genuinely hard to detect at this n, so treat SRM as a screen rather than a guarantee.

**End-to-end readout** (simulated "checkout redesign", n = 40,000, seed 7, true average primary effect about +0.275 with segment effects): the primary effect was +0.317, 95% CI [0.250, 0.385] with CUPED (24.7% variance reduction), and SRM p = 0.418. mSPRT rejected at n = 4,000 of 40,000. Segments A and B were significant after Holm correction, with Cochran Q p = 5e-7. Losing 3% of treated users in the same data trips the SRM check (p = 1.3e-4) and the readout is marked `invalid_srm`.

## Limitations

* Two-arm experiments only (control + one treatment). Multi-arm designs need a family-wise adjustment over arms.
* Randomisation is at the unit level with independent units. Cluster/network interference and switchback designs are not covered.
* The service keeps state in memory (one process). Swap `Store` for a database to persist it. There is no authentication.
* The sequential simulations assume normal data. With heavy tails and small n, the plug-in variance is only asymptotically valid.
* The CATE T-learner validates ranking quality. It does not establish causal identification beyond randomisation. EconML is optional.

## License

MIT
