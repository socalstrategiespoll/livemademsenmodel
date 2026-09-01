# Render web service: polls civicAPI on a background thread and serves the
# projection for the MA Senate Democratic Primary (Markey vs. Moulton).
#
# NOTE ON REUSE: the Wisconsin/Michigan server.py files were written against a
# model class with hardcoded hong/crowley/other fields (c.hong_votes,
# model.project_county(), etc.). This file is written against the GENERIC
# election_model_template.py architecture instead -- dict-based CountyState,
# CANDIDATES-driven -- because that's what ma_senate_model.py uses. If you're
# starting a future race from THIS file, it should carry over close to as-is
# (just change the import and BASELINE_PATH default); if you're starting from
# the Wisconsin/Michigan server.py files instead, expect a real rewrite, not a
# "few line" change, since their build_output/build_county_table are written
# against named per-candidate fields that don't exist on the generic model.
#
# Same deploy reasoning as the WI/MI builds: a persistent web service (not a
# cron job) because Render destroys cron containers after every run, which
# would wipe the town-level turnout/shift state this model accumulates, and a
# cron job has no URL for the site to read.

import json
import os
import threading
import time
import traceback

import numpy as np

from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from ma_senate_model import ElectionModel, CANDIDATES
from ma_civicapi_feed import fetch_race, parse_payload, MA_SENATE_DEM_PRIMARY


PORT = int(os.environ.get("PORT", 10000))
RACE_ID = os.environ.get("RACE_ID") or MA_SENATE_DEM_PRIMARY
N_SIMS = int(os.environ.get("N_SIMS", 20000))
POLL_INTERVAL = int(os.environ.get("POLL_INTERVAL", 60))
HISTORY_LIMIT = int(os.environ.get("HISTORY_LIMIT", 2000))
STATE_DIR = os.environ.get("STATE_DIR", "")
BASELINE_PATH = os.environ.get("BASELINE_PATH", "ma_senate_baselines.json")

A, B = CANDIDATES[0], CANDIDATES[1]  # markey, moulton


class ModelState:
    """Everything the poller produces and the HTTP handler reads."""

    def __init__(self):
        self.lock = threading.Lock()
        self.projection = None
        self.history = []
        self.error = None
        self.cycles = 0
        self.started_at = datetime.now(timezone.utc).isoformat()

    def publish(self, output: dict) -> None:
        with self.lock:
            self.projection = output
            self.history.append({
                "updated_at": output["updated_at"],
                f"{A}_pct": output["projection"][f"{A}_pct"],
                f"{B}_pct": output["projection"][f"{B}_pct"],
                f"{A}_win_probability": output["projection"][f"{A}_win_probability"],
                "interval_90": output["projection"]["interval_90"],
                "pct_counted": output["counted"]["pct_of_projected_turnout"],
                "towns_reporting": output["diagnostics"]["towns_reporting"],
                "statewide_shift": output["diagnostics"]["statewide_shift"],
            })
            if len(self.history) > HISTORY_LIMIT:
                self.history = self.history[-HISTORY_LIMIT:]
            self.error = None
            self.cycles += 1

    def fail(self, message: str) -> None:
        with self.lock:
            self.error = message

    def snapshot(self) -> tuple:
        with self.lock:
            return self.projection, list(self.history), self.error, self.cycles


STATE = ModelState()


def build_output(model: ElectionModel, sim: dict, proj: dict,
                  parsed: dict, race_id) -> dict:
    total_expected = sum(c.effective_turnout for c in model.counties.values())
    counted_actual = sum(c.counted_votes for c in model.counties.values())

    projection = {
        f"{A}_win_probability": round(sim[f"{A}_win_prob"], 4),
        f"{B}_win_probability": round(sim[f"{B}_win_prob"], 4),
        "median_margin": round(sim["p50"], 2),
        # Genuine middle-50/middle-90 intervals -- p25/p75 and p05/p95
        # respectively. See ma_senate_model.py's module docstring: the
        # Wisconsin reference build shipped a bug where "interval_50" was
        # actually p10/p90 (an 80% interval) mislabeled. Verified NOT
        # reproduced here -- these are the real percentiles, not a renamed
        # copy of the 90% band.
        "interval_50": [round(sim["p25"], 2), round(sim["p75"], 2)],
        "interval_90": [round(sim["p05"], 2), round(sim["p95"], 2)],
        f"{A}_votes": int(proj[f"{A}_votes"]),
        f"{B}_votes": int(proj[f"{B}_votes"]),
        # Raw simulated distribution, thinned to ~200 percentiles (not 60 --
        # see UNIVERSAL_TEMPLATE_GUIDE.md debugging lesson 7, too few points
        # makes the density chart spiky regardless of model correctness).
        "margin_percentiles": [
            round(float(v), 2) for v in
            np.percentile(sim["margins"], np.arange(0.25, 100, 0.5))
        ],
        "share_ranges": {
            cand: {
                "median": round(r["p50"], 2),
                "range_50": [round(r["p25"], 2), round(r["p75"], 2)],
                "range_90": [round(r["p05"], 2), round(r["p95"], 2)],
            }
            for cand, r in sim["candidate_share_ranges"].items()
        },
    }
    for cand in CANDIDATES:
        projection[f"{cand}_pct"] = round(proj[f"{cand}_pct"], 2)

    return {
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "source": "civicapi.org",
        "attribution": "Election results from civicAPI (civicapi.org)",
        "race_id": race_id,
        "election_name": parsed.get("election_name"),
        "feed_last_updated": parsed.get("last_updated"),
        "counted": {
            A: parsed.get(f"state_{A}"),
            B: parsed.get(f"state_{B}"),
            "dropped_other_candidates": parsed.get("state_dropped_other_candidates"),
            "pct_of_projected_turnout": round(100 * counted_actual / max(total_expected, 1), 2),
            "pct_precincts_reporting": parsed.get("percent_precincts_statewide"),
        },
        "turnout": {"projected": round(total_expected)},
        "projection": projection,
        "towns": build_town_table(model),
        "diagnostics": {
            "towns_reporting": sum(1 for c in model.counties.values() if c.pct_reporting > 0),
            "statewide_shift": round(model.statewide_shift[A] - model.statewide_shift[B], 2),
            "statewide_shift_by_candidate": {k: round(v, 2) for k, v in model.statewide_shift.items()},
            "unmatched_towns": parsed.get("unmatched", []),
            "candidate_names": parsed.get("candidate_names"),
        },
        "regional_shift": {
            region: round(model.regional_shift[A][region] - model.regional_shift[B][region], 2)
            for region in model.regional_shift[A]
        },
    }


def build_town_table(model: ElectionModel) -> list:
    """Per-town rows covering ALL 348 towns, not just the ones reporting --
    the maps need every town every cycle. Margin convention: A (markey) minus
    B (moulton), as a share of the two-way total (there's no third bucket to
    fold in here, unlike a 3-candidate race)."""
    rows = []
    for name, c in model.counties.items():
        va, vb = c.votes.get(A, 0), c.votes.get(B, 0)
        margin = c.two_way_margin(A, B)
        baseline_margin = c.baseline_two_way_margin(A, B)
        proj_margin_two_way = 100.0 * (model.project_rate(c, A) - model.project_rate(c, B)) / (
            model.project_rate(c, A) + model.project_rate(c, B))
        remaining = max(0, c.effective_turnout - c.counted_votes)

        rows.append({
            "town": name,
            "region": c.region,
            "reporting": c.pct_reporting > 0,
            A: va,
            B: vb,
            "votes": c.counted_votes,
            "margin": margin,
            "expected_baseline": round(baseline_margin, 1),
            "vs_expected": None if margin is None else round(margin - baseline_margin, 1),
            "town_shift": round(
                model.county_shift[A].get(name, 0.0) - model.county_shift[B].get(name, 0.0), 1),
            "pct_precincts": c.pct_reporting * 100 if c.pct_reporting else None,
            "pct_of_projected": round(100 * c.counted_votes / max(c.effective_turnout, 1), 1),
            "projected_total": int(c.effective_turnout),
            "calibrated_turnout": int(c.calibrated_turnout) if c.calibrated_turnout else None,
            "remaining": int(round(remaining)),
            "remainder_margin": round(proj_margin_two_way, 1),
            "projected_final": round(proj_margin_two_way, 1),
        })

    rows.sort(key=lambda r: (-r["votes"], -r["projected_total"]))
    return rows


def save_state(model: ElectionModel) -> None:
    if not STATE_DIR:
        return
    try:
        os.makedirs(STATE_DIR, exist_ok=True)
        snap = {
            name: {
                "votes": {cand: c.votes.get(cand, 0) for cand in CANDIDATES},
                "pct_reporting": c.pct_reporting,
            }
            for name, c in model.counties.items() if c.pct_reporting > 0
        }
        with open(os.path.join(STATE_DIR, "feed_state.json"), "w") as handle:
            json.dump(snap, handle)
    except Exception:
        pass


def load_state(model: ElectionModel) -> None:
    if not STATE_DIR:
        return
    path = os.path.join(STATE_DIR, "feed_state.json")
    try:
        with open(path) as handle:
            stored = json.load(handle)
        for name, rec in stored.items():
            if name in model.counties:
                model.update_county(name, rec["votes"], rec["pct_reporting"])
        print("restored {} towns from {}".format(len(stored), path), flush=True)
    except Exception:
        pass


def poller() -> None:
    """Background loop. Never exits."""
    model = ElectionModel(BASELINE_PATH)
    load_state(model)
    town_names = list(model.counties.keys())

    print("poller started: race {} every {}s, {} sims".format(
        RACE_ID, POLL_INTERVAL, N_SIMS), flush=True)

    while True:
        started = time.time()
        try:
            payload = fetch_race(RACE_ID)
            parsed = parse_payload(payload, town_names)

            for town, record in parsed["towns"].items():
                pct = record.get("percent_precincts") or 0.0
                pct = pct / 100.0 if pct > 1 else pct
                model.update_county(town, {A: record[A], B: record[B]}, pct)

            sim = model.run_simulation(n_sims=N_SIMS)
            proj = model.statewide_projection()
            output = build_output(model, sim, proj, parsed, RACE_ID)
            STATE.publish(output)
            save_state(model)

            names = output["diagnostics"].get("candidate_names") or {}
            if not names.get(A) or not names.get(B):
                print("!! CANDIDATE MATCH FAILED: {}={!r} {}={!r} -- fix "
                      "MARKEY_KEYS / MOULTON_KEYS in ma_civicapi_feed.py".format(
                          A, names.get(A), B, names.get(B)), flush=True)
            else:
                print("   matched: {} vs {}".format(names[A], names[B]), flush=True)
            if output["diagnostics"]["unmatched_towns"]:
                print("!! UNMATCHED TOWNS: {} -- fix normalize_town() in "
                      "ma_civicapi_feed.py".format(
                          output["diagnostics"]["unmatched_towns"]), flush=True)

            p = output["projection"]
            print("[{}] {:.1f}% counted | {} towns | {} {:.1f}  {} {:.1f} | "
                  "margin {:+.1f} [{:+.1f}, {:+.1f}] | {} win {:.1%}".format(
                      datetime.now().strftime("%H:%M:%S"),
                      output["counted"]["pct_of_projected_turnout"],
                      output["diagnostics"]["towns_reporting"],
                      A, p[f"{A}_pct"], B, p[f"{B}_pct"], p["median_margin"],
                      p["interval_90"][0], p["interval_90"][1],
                      A, p[f"{A}_win_probability"]), flush=True)

        except Exception as exc:
            STATE.fail(str(exc))
            print("[{}] cycle failed, serving last good projection: {}".format(
                datetime.now().strftime("%H:%M:%S"), exc), flush=True)
            traceback.print_exc()

        time.sleep(max(1.0, POLL_INTERVAL - (time.time() - started)))


class Handler(BaseHTTPRequestHandler):

    def _send(self, body, status=200, content_type="application/json"):
        encoded = (body if isinstance(body, bytes) else json.dumps(body).encode("utf-8"))
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(encoded)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(encoded)

    def do_OPTIONS(self):
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_GET(self):
        path = self.path.split("?")[0].rstrip("/") or "/"
        projection, history, error, cycles = STATE.snapshot()

        if path in ("/", "/health"):
            return self._send({
                "ok": True, "cycles": cycles, "started_at": STATE.started_at,
                "last_error": error, "has_projection": projection is not None,
            })
        if path == "/api/projection":
            if projection is None:
                return self._send({"error": "no projection yet", "last_error": error}, status=503)
            return self._send(projection)
        if path == "/api/history":
            return self._send({"count": len(history), "cycles": history})
        return self._send({"error": "not found"}, status=404)

    def log_message(self, *args):
        return


def main():
    thread = threading.Thread(target=poller, daemon=True)
    thread.start()
    server = ThreadingHTTPServer(("0.0.0.0", PORT), Handler)
    print("serving on :{}".format(PORT), flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
