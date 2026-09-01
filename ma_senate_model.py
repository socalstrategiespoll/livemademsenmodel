"""
Massachusetts U.S. Senate Democratic Primary — live election-night model.
Markey vs. Moulton, 2-candidate deductive/independent-candidate architecture,
built from the Wisconsin Governor Democratic Primary template
(election_model_template.py / UNIVERSAL_TEMPLATE_GUIDE.md).

GEOGRAPHIC UNIT: towns/cities, not counties. Massachusetts elections report at
the town/city level (351 municipalities), and county government is largely
vestigial here, so every "county" in the generic template's variable names
below is actually a town. The engine doesn't care about the label -- it just
needs a name, a region, a baseline, and a turnout figure per unit.

BASELINE CONSTRUCTION (see ma_senate_baselines.json + build_baselines.py):
  1. Source data: 2020 Markey vs. Kennedy Democratic Senate primary results,
     348 of MA's 351 towns (Franklin was missing from the pasted dataset).
  2. Home-district adjustment: Kennedy represented the pre-2021-redistricting
     MA-4 (Newton/Brookline/Wellesley down through Fall River/Taunton/
     Attleboro, 34 of 35 towns present in this dataset). His 2020 numbers in
     those towns carry an incumbent home-turf bump on top of the underlying
     demographic lean. Estimated via region-matched comparison (CD4 towns vs.
     non-CD4 towns of the same character -- SE Mass working-class cities,
     affluent Norfolk/Middlesex suburbs, and Blackstone Valley mill towns
     scored separately): +4.9, +3.9, and +4.0 points respectively, a tight
     enough spread across three very different regional contexts to trust as
     a real effect rather than noise (a first pass using a simple log-turnout
     regression gave an unreliable, noisy estimate -- weighted R^2 of 0.04 --
     and was discarded in favor of this region-matched comparison). Turnout-
     weighted average: +4.26 points, removed from the Kennedy-derived
     ("Moulton proxy") share in every CD4 town before anything else.
  3. Moulton's own boost: the same +4.26 points added to the Moulton-proxy
     share in every town in his current MA-6 (2021 lines, 39 towns, Amesbury
     through Wilmington). This is a symmetry assumption (same magnitude as
     the Kennedy effect), not independently estimated -- revisit if better
     information on Moulton's personal-vote strength becomes available.
  4. Calibration to target: after the above adjustments, the turnout-weighted
     statewide average landed at Markey 55.33 / Moulton 44.67. A single
     uniform +6.67-point additive shift was applied to every town's Markey
     share (and correspondingly subtracted from Moulton) to hit the target
     topline of Markey 62 / Moulton 38 exactly, preserving each town's
     relative geographic variation rather than hand-adjusting individual
     towns (per the guide's calibration methodology).
  5. Turnout: 2020's 1,383,195 scaled down by a uniform factor to a 800,000
     statewide total, preserving each town's relative share.

COALITION-KERNEL PROXY: coalition_index.json, a per-town index of the town's
2020 Kennedy share relative to the 2020 statewide Kennedy share (1.0 = exactly
average). Thematically apt for this race: Kennedy, like Moulton, drew a more
moderate/establishment-aligned coalition against a more progressive incumbent,
so a town's relative Kennedy-strength in 2020 is a reasonable similarity
signal for how it might swing between Markey and Moulton now. Same role as
Wisconsin's 2016 Clinton/Sanders index.

CANDIDATES = ("markey", "moulton") -- no "Other" bucket, per Wilson's
instruction to set the statewide baseline as Markey 62% / Moulton 38% (a
clean two-candidate framing). If minor candidates end up qualifying for the
2026 ballot and start polling non-trivially, add a third "other" key here and
in the baseline JSON -- the engine adapts automatically, nothing else changes.

CONFIDENCE-INTERVAL FIX (the specific ask that started this build): the
Wisconsin reference build had a real bug where "Middle 50%" was computed from
p10/p90 (an 80% interval) instead of p25/p75, caught by comparing the
displayed range against the win probability shown next to it. This template
file already carries the fix -- see run_simulation() below, where p25/p75 are
the genuine middle-50 bound and p10/p90 are reported separately as their own
labeled interval, not reused under the wrong name. Nothing about the fix is
race-specific; it just needed to be verified as present, not reintroduced.
"""

import json
import math
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, Optional

# ------------------------------------------------------------------
CANDIDATES = ("markey", "moulton")

# ------------------------------------------------------------------
# Tunable constants -- these worked well as-is for a 72-county, 3-candidate
# race and are not principled functions of candidate count, so leave them
# unless your state's county count or counting patterns are unusual.
# ------------------------------------------------------------------
CREDIBILITY_EXPONENT = 2.0
OUTLIER_LAMBDA = 3.0
TAU_FLOOR = 0.08
N_SIMS = 20000

TURNOUT_FULL_TRUST_PCT = 0.25
TURNOUT_CLAMP = (0.40, 2.50)

MOMENTUM_TRIGGER_PCT = 0.30
MOMENTUM_MAX_DRIFT = 10.0

GLOBAL_EVIDENCE_PRIOR = 500.0
REGIONAL_EVIDENCE_PRIOR = 50.0
KERNEL_EVIDENCE_PRIOR = 50.0

# Both candidates are well-known, well-polled statewide figures (incumbent
# senator vs. sitting congressman) -- no aggregate "Other" bucket with extra
# uncertainty here, so one shared default is reasonable. This is a STARTING
# POINT: retune via run_simulation() once there's an actual pre-election poll
# to target a specific underdog win probability against; 9.0 is carried over
# unchanged from the Wisconsin build and hasn't been validated for this race.
PRE_ELECTION_SD = {
    "default": 9.0,
}

# This scale factor converts OBSERVED cross-county disagreement
# (statewide_shift_var, real data once results start coming in) into margin-
# point uncertainty. Unlike PRE_ELECTION_SD, this matters most AFTER counting
# starts, not before -- if your confidence interval looks too wide once real
# results are flowing in, THIS is very likely the lever to retune, not the
# pre-election prior (which is already mostly shrunk out by then). Tested
# range in production: values from 4 to 15 all "worked" in the sense of not
# crashing, but produced meaningfully different interval widths -- there's no
# principled way to derive this number, only to tune it against how wide the
# live interval looks once real disagreement starts showing up. See the
# debugging-lessons section at the bottom of the guide for the full story.
STATEWIDE_SHIFT_SCALE = 4.0

# per-county remainder uncertainty for votes not yet counted. Matters most
# EARLY (before much real disagreement has been observed) and matters little
# once STATEWIDE_SHIFT_SCALE's term dominates -- don't expect this alone to
# fix a too-wide live interval; test both before concluding this is the lever.
BASE_REMAINDER_SD = 8.0

# Towns large/internally-diverse enough that a partial count is a biased draw
# of the town, not a random sample of it -- Boston most of all (a very large,
# demographically varied electorate reporting ward-by-ward), plus the state's
# next two largest cities. Worth revisiting once real 2026 polling shows
# whether any other city is unusually split internally on Markey vs. Moulton.
COUNTY_HETEROGENEITY = {
    "DEFAULT": 2.5,
    "Boston": 10.0,
    "Worcester": 5.0,
    "Springfield": 5.0,
}

# 9 geographic regions covering all 348 towns in the baseline data (Franklin
# was missing from the source dataset and isn't modeled). See regions.py for
# the full town lists and build_baselines.py for how they were assigned --
# roughly media-market/character based (Boston core, inner suburbs, MetroWest,
# North Shore/Essex, Merrimack Valley, South Shore/Plymouth, SE Mass/Bristol,
# Central Mass/Worcester, Western Mass, Cape & Islands), not strictly county
# lines, since Wilson confirmed town-level baselines don't need county
# aggregation for this build.
with open("ma_senate_regions.json") as _f:
    REGIONS: Dict[str, list] = json.load(_f)
COUNTY_REGION = {c: r for r, cs in REGIONS.items() for c in cs}

# Coalition-similarity kernel: each town's 2020 Kennedy share relative to the
# 2020 statewide Kennedy share (1.0 = exactly average). See module docstring
# for why this is a reasonable proxy for a Markey/Moulton coalition split.
with open("coalition_index.json") as _f:
    COALITION_PROXY_INDEX: Dict[str, float] = json.load(_f)
DEFAULT_COALITION_INDEX = 1.0


@dataclass
class CountyState:
    name: str
    region: str
    baseline_pct: Dict[str, float]      # {candidate: baseline % of full electorate}
    expected_turnout: int               # ORIGINAL prior turnout, never mutated
    calibrated_turnout: Optional[float] = None
    pct_reporting: float = 0.0
    counted_votes: int = 0
    votes: Dict[str, int] = field(default_factory=dict)               # {candidate: votes counted}
    observed_rate_: Dict[str, Optional[float]] = field(default_factory=dict)  # {candidate: votes/counted_votes}

    @property
    def effective_turnout(self) -> float:
        return self.calibrated_turnout if self.calibrated_turnout is not None else self.expected_turnout

    def baseline_rate(self, candidate: str) -> float:
        return self.baseline_pct[candidate] / 100.0

    def observed_rate(self, candidate: str) -> Optional[float]:
        return self.observed_rate_.get(candidate)

    @property
    def heterogeneity(self) -> float:
        return COUNTY_HETEROGENEITY.get(self.name, COUNTY_HETEROGENEITY["DEFAULT"])

    @property
    def credibility(self) -> float:
        if self.pct_reporting <= 0:
            return 0.0
        completeness_weight = self.pct_reporting ** (1 / CREDIBILITY_EXPONENT)
        design_var = (self.heterogeneity ** 2) * (1 - self.pct_reporting)
        noise_penalty = 1.0 / (1.0 + design_var / 50.0)
        return completeness_weight * noise_penalty

    @property
    def coalition_index(self) -> float:
        return COALITION_PROXY_INDEX.get(self.name, DEFAULT_COALITION_INDEX)

    def two_way_margin(self, cand_a: str, cand_b: str) -> Optional[float]:
        """Convenience: margin between any two tracked candidates among counted
        votes, e.g. for a display headline. Not used internally by the engine."""
        va, vb = self.votes.get(cand_a, 0), self.votes.get(cand_b, 0)
        if va + vb <= 0:
            return None
        return 100.0 * (va - vb) / (va + vb)

    def baseline_two_way_margin(self, cand_a: str, cand_b: str) -> float:
        a, b = self.baseline_pct[cand_a], self.baseline_pct[cand_b]
        if a + b <= 0:
            return 0.0
        return 100.0 * (a - b) / (a + b)


class ElectionModel:
    def __init__(self, baseline_path: str):
        with open(baseline_path) as f:
            baselines = json.load(f)
        self.counties: Dict[str, CountyState] = {}
        for name, b in baselines.items():
            self.counties[name] = CountyState(
                name=name, region=b["region"],
                baseline_pct={k: b[k] for k in CANDIDATES},
                expected_turnout=b["turnout"],
            )
        self.total_evidence_weight = 0.0
        self.statewide_shift: Dict[str, float] = {k: 0.0 for k in CANDIDATES}
        self.statewide_shift_var: Dict[str, float] = {k: TAU_FLOOR ** 2 for k in CANDIDATES}
        self.regional_shift: Dict[str, Dict[str, float]] = {k: {r: 0.0 for r in REGIONS} for k in CANDIDATES}
        self.county_shift: Dict[str, Dict[str, float]] = {k: {c: 0.0 for c in self.counties} for k in CANDIDATES}

    # ------------------------------------------------------------
    def update_county(self, name: str, votes: Dict[str, int], pct_reporting: float):
        """votes: {candidate: vote_count} for ALL candidates in CANDIDATES.
        Missing keys are treated as 0."""
        c = self.counties[name]
        c.votes = {k: votes.get(k, 0) for k in CANDIDATES}
        c.counted_votes = sum(c.votes.values())
        c.pct_reporting = pct_reporting
        if c.counted_votes > 0:
            c.observed_rate_ = {k: c.votes[k] / c.counted_votes for k in CANDIDATES}
        self._recalibrate_turnout()
        self._recompute_shifts()

    # ------------------------------------------------------------
    def _recalibrate_turnout(self):
        """Feed-implied turnout (counted/pct_reporting) replaces the static prior
        wherever a county has enough reporting to trust it, credibility-ramped
        and clamped. Counties still at 0% get the size-weighted median ratio
        from reporting counties."""
        ratios, sizes = [], []
        for c in self.counties.values():
            if c.pct_reporting > 0 and c.counted_votes > 0:
                implied = c.counted_votes / c.pct_reporting
                ratio = implied / c.expected_turnout
                ratio = min(max(ratio, TURNOUT_CLAMP[0]), TURNOUT_CLAMP[1])
                trust = min(c.pct_reporting / TURNOUT_FULL_TRUST_PCT, 1.0)
                c.calibrated_turnout = trust * (ratio * c.expected_turnout) + (1 - trust) * c.expected_turnout
                ratios.append(ratio)
                sizes.append(c.expected_turnout)

        if not ratios:
            return
        ratios = np.array(ratios)
        sizes = np.array(sizes)
        order = np.argsort(ratios)
        cum_size = np.cumsum(sizes[order])
        median_idx = np.searchsorted(cum_size, cum_size[-1] / 2.0)
        size_weighted_median_ratio = ratios[order][min(median_idx, len(ratios) - 1)]

        for c in self.counties.values():
            if c.pct_reporting == 0:
                c.calibrated_turnout = size_weighted_median_ratio * c.expected_turnout

    # ------------------------------------------------------------
    # Hierarchical shift: universal + regional + coalition kernel, computed
    # independently for EACH tracked candidate.
    # ------------------------------------------------------------
    KERNEL_BANDWIDTH = 0.12

    def _recompute_shifts(self):
        reporting = [c for c in self.counties.values() if c.pct_reporting > 0 and c.counted_votes > 0]
        if not reporting:
            self.total_evidence_weight = 0.0
            for k in CANDIDATES:
                self.statewide_shift[k] = 0.0
                self.regional_shift[k] = {r: 0.0 for r in REGIONS}
                self.county_shift[k] = {c: 0.0 for c in self.counties}
            return

        regions = np.array([c.region for c in reporting])
        coalition_idx = np.array([c.coalition_index for c in reporting])
        turnouts = np.array([c.effective_turnout for c in reporting])
        credibilities = np.array([c.credibility for c in reporting])
        base_w = credibilities * np.sqrt(turnouts)

        total_weight_for_evidence = 0.0

        for candidate in CANDIDATES:
            surprises = np.array([
                100.0 * (c.observed_rate(candidate) - c.baseline_rate(candidate)) for c in reporting
            ])
            outlier_factor = 1.0 / (1.0 + (np.abs(surprises) / OUTLIER_LAMBDA) ** 2)
            weights = base_w * outlier_factor
            total_weight_for_evidence = max(total_weight_for_evidence, float(weights.sum()))

            total_weight = weights.sum()
            if total_weight == 0:
                self.statewide_shift[candidate] = 0.0
            else:
                wmean = np.average(surprises, weights=weights)
                tau2 = max(TAU_FLOOR ** 2, np.average((surprises - wmean) ** 2, weights=weights))
                global_shrink = total_weight / (total_weight + GLOBAL_EVIDENCE_PRIOR)
                self.statewide_shift[candidate] = global_shrink * wmean
                self.statewide_shift_var[candidate] = tau2

            for region in REGIONS:
                idx = regions == region
                if not idx.any():
                    self.regional_shift[candidate][region] = self.statewide_shift[candidate]
                    continue
                r_wmean = (np.average(surprises[idx], weights=weights[idx])
                           if weights[idx].sum() > 0 else self.statewide_shift[candidate])
                shrink = weights[idx].sum() / (weights[idx].sum() + REGIONAL_EVIDENCE_PRIOR)
                self.regional_shift[candidate][region] = (
                    shrink * r_wmean + (1 - shrink) * self.statewide_shift[candidate])

            for name, county in self.counties.items():
                kernel_w = weights * np.exp(-((coalition_idx - county.coalition_index) ** 2) / (2 * self.KERNEL_BANDWIDTH ** 2))
                kernel_w = kernel_w * np.where(regions == county.region, 1.5, 1.0)
                if kernel_w.sum() <= 0:
                    local_est = self.regional_shift[candidate][county.region]
                else:
                    local_est = np.average(surprises, weights=kernel_w)
                shrink = kernel_w.sum() / (kernel_w.sum() + KERNEL_EVIDENCE_PRIOR)
                self.county_shift[candidate][name] = (
                    shrink * local_est + (1 - shrink) * self.regional_shift[candidate][county.region])

        self.total_evidence_weight = total_weight_for_evidence

    # ------------------------------------------------------------
    def project_rate(self, c: CountyState, candidate: str) -> float:
        """Projected share of the vote for one candidate in one county, as a
        fraction of the full electorate (0-1). Independent per candidate;
        normalize across CANDIDATES wherever votes are actually allocated."""
        baseline_rate = c.baseline_rate(candidate)
        shift = self.county_shift[candidate].get(c.name, 0.0) / 100.0
        adjusted_baseline = min(max(baseline_rate + shift, 0.0), 0.97)

        observed = c.observed_rate(candidate)
        if c.pct_reporting >= 0.999:
            return observed if observed is not None else adjusted_baseline
        if observed is None:
            return adjusted_baseline

        w = c.credibility
        projected = w * observed + (1 - w) * adjusted_baseline

        if c.pct_reporting >= MOMENTUM_TRIGGER_PCT:
            lo = observed - MOMENTUM_MAX_DRIFT / 100.0
            hi = observed + MOMENTUM_MAX_DRIFT / 100.0
            projected = min(max(projected, lo), hi)

        return min(max(projected, 0.0), 0.97)

    # ------------------------------------------------------------
    def statewide_projection(self) -> Dict[str, float]:
        totals = {k: 0.0 for k in CANDIDATES}
        for c in self.counties.values():
            remaining_votes = max(0, c.effective_turnout - c.counted_votes)
            raw = {k: self.project_rate(c, k) for k in CANDIDATES}
            raw_total = sum(raw.values())
            if raw_total <= 0:
                shares = {k: 1.0 / len(CANDIDATES) for k in CANDIDATES}
            else:
                shares = {k: raw[k] / raw_total for k in CANDIDATES}
            for k in CANDIDATES:
                totals[k] += c.votes.get(k, 0) + remaining_votes * shares[k]

        grand_total = sum(totals.values())
        result = {f"{k}_pct": 100 * totals[k] / grand_total for k in CANDIDATES}
        result.update({f"{k}_votes": totals[k] for k in CANDIDATES})
        if len(CANDIDATES) >= 2:
            a, b = CANDIDATES[0], CANDIDATES[1]
            result["statewide_shift"] = self.statewide_shift[a] - self.statewide_shift[b]
        return result

    # ------------------------------------------------------------
    def run_simulation(self, n_sims: int = N_SIMS, seed: Optional[int] = None) -> Dict:
        """Vectorized Monte Carlo: each tracked candidate gets its own shared
        statewide shock + per-county shock, drawn independently, then all
        candidates' simulated rates are normalized to sum to 1 per-simulation
        before allocating the remaining vote. Margin (for len(CANDIDATES)>=2)
        is reported as CANDIDATES[0] minus CANDIDATES[1], as a share of the
        FULL electorate (every candidate in the denominator)."""
        rng = np.random.default_rng(seed)
        counties = list(self.counties.values())
        n = len(counties)

        completeness = np.array([c.pct_reporting for c in counties])
        heterog = np.array([c.heterogeneity for c in counties])
        eff_turnout = np.array([c.effective_turnout for c in counties])
        counted = np.array([c.counted_votes for c in counties])
        remaining_votes = np.maximum(0, eff_turnout - counted)

        county_sd = BASE_REMAINDER_SD * (1 - completeness) ** 0.5 + heterog * (1 - completeness) * 0.3
        county_sd = np.maximum(county_sd, 0.5)

        evidence_shrink = self.total_evidence_weight / (self.total_evidence_weight + GLOBAL_EVIDENCE_PRIOR)
        n_cand = len(CANDIDATES)
        prior_sd = {}
        for k in CANDIDATES:
            target = PRE_ELECTION_SD.get(k, PRE_ELECTION_SD["default"])
            # divide by sqrt(n_cand) as a STARTING POINT for this candidate's
            # share of a target aggregate-margin SD -- not exact once momentum
            # constraints and normalization (both nonlinear) are involved,
            # retune empirically against run_simulation()'s actual output
            prior_sd[k] = (target / math.sqrt(n_cand)) * (1 - evidence_shrink)

        sim_rates = {}
        actual_votes = {}
        for candidate in CANDIDATES:
            point_rate = np.array([self.project_rate(c, candidate) for c in counties])
            statewide_sd = math.sqrt(self.statewide_shift_var[candidate]) * STATEWIDE_SHIFT_SCALE
            statewide_sd = math.sqrt(statewide_sd ** 2 + prior_sd[candidate] ** 2)

            momentum_active = np.array([
                c.pct_reporting >= MOMENTUM_TRIGGER_PCT and c.observed_rate(candidate) is not None
                for c in counties
            ])
            obs_arr = np.array([
                (c.observed_rate(candidate) if c.observed_rate(candidate) is not None else 0.0)
                for c in counties
            ])
            lo_bound = obs_arr - MOMENTUM_MAX_DRIFT / 100.0
            hi_bound = obs_arr + MOMENTUM_MAX_DRIFT / 100.0

            shared_shock = rng.normal(0, statewide_sd, size=(n_sims, 1)) / 100.0
            county_shock = rng.normal(0, 1, size=(n_sims, n)) * (county_sd[None, :] / 100.0)
            sim_rate = point_rate[None, :] + shared_shock + county_shock

            clipped = np.clip(sim_rate, lo_bound[None, :], hi_bound[None, :])
            sim_rate = np.where(momentum_active[None, :], clipped, sim_rate)
            sim_rates[candidate] = np.clip(sim_rate, 0.0, 0.97)
            actual_votes[candidate] = np.array([c.votes.get(candidate, 0) for c in counties], dtype=float)

        raw_total = sum(sim_rates.values())
        raw_total = np.maximum(raw_total, 1e-9)
        shares = {k: sim_rates[k] / raw_total for k in CANDIDATES}

        candidate_totals = {
            k: (actual_votes[k][None, :] + remaining_votes[None, :] * shares[k]).sum(axis=1)
            for k in CANDIDATES
        }
        grand_totals = sum(candidate_totals.values())

        def pct_range(arr):
            return {
                "p05": float(np.percentile(arr, 5)), "p25": float(np.percentile(arr, 25)),
                "p50": float(np.percentile(arr, 50)),
                "p75": float(np.percentile(arr, 75)), "p95": float(np.percentile(arr, 95)),
            }

        # Each candidate's OWN simulated statewide vote-share distribution
        # (not the margin between two of them) -- useful for showing a range
        # per candidate, not just the leader-vs-second-place margin.
        candidate_share_ranges = {
            k: pct_range(100 * candidate_totals[k] / grand_totals) for k in CANDIDATES
        }

        out = {"n_sims": n_sims, "candidate_share_ranges": candidate_share_ranges}
        if len(CANDIDATES) >= 2:
            a, b = CANDIDATES[0], CANDIDATES[1]
            # Margin as share of the FULL electorate (every candidate in the
            # denominator), not normalized to just the a/b two-way pool.
            results = 100 * candidate_totals[a] / grand_totals - 100 * candidate_totals[b] / grand_totals
            out.update({
                "mean_margin": float(np.mean(results)),
                # NOTE: p25/p75 are the genuine "middle 50%" -- a longstanding
                # bug in the reference build used p10/p90 (an 80% interval)
                # mislabeled as "Middle 50%" for months before it was caught
                # by comparing displayed percentiles against win probability
                # and finding them inconsistent. Don't repeat that: if you
                # display a "middle X%" range, verify which percentiles you
                # actually plugged in, don't assume the variable name is honest.
                "p05": float(np.percentile(results, 5)),
                "p10": float(np.percentile(results, 10)),
                "p25": float(np.percentile(results, 25)),
                "p50": float(np.percentile(results, 50)),
                "p75": float(np.percentile(results, 75)),
                "p90": float(np.percentile(results, 90)),
                "p95": float(np.percentile(results, 95)),
                f"{a}_win_prob": float(np.mean(results > 0)),
                f"{b}_win_prob": float(np.mean(results < 0)),
                "margins": results,
            })
        return out
