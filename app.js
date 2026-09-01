// ---------------------------------------------------------------------------
// Point this at your Render service URL. No trailing slash.
// Example: https://wi-governor-primary-model.onrender.com
// ---------------------------------------------------------------------------
const API_BASE = "https://ma-senate-primary-model.onrender.com"; // update after Render deploy

const REFRESH_MS = 15000;
const STALE_AFTER_MS = 180000;

const MARKEY = "Markey";
const MOULTON = "Moulton";

const num = new Intl.NumberFormat("en-US");
const $ = (id) => document.getElementById(id);
const reduceMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;

function signed(v, d = 1) {
  return (v >= 0 ? "+" : "\u2212") + Math.abs(v).toFixed(d);
}

// ---------------------------------------------------------------------------
// Distribution -- identical mechanism to the MI build, drawn from the
// simulated percentiles the model actually produced.
// ---------------------------------------------------------------------------

const W = 720;
const H = 200;
const PAD = 8;

function density(percentiles, bins = 90) {
  const lo = percentiles[0];
  const hi = percentiles[percentiles.length - 1];
  const span = hi - lo || 1;

  // The input is already a percentile array (evenly spaced in PROBABILITY,
  // not raw samples), so a naive histogram double-bins it -- spikes appear
  // wherever percentiles happen to cluster tightly in a given bin, and the
  // sparse tails create jagged edge noise. Instead, compute density directly
  // as the derivative of the percentile function: how much probability mass
  // sits between each pair of consecutive percentile points, divided by how
  // far apart those points are in value. This is smooth by construction.
  const n = percentiles.length;
  const stepProb = 100 / (n - 1);
  const midX = [];
  const rawDensity = [];
  for (let i = 0; i < n - 1; i++) {
    const dv = Math.max(percentiles[i + 1] - percentiles[i], 1e-6);
    midX.push((percentiles[i] + percentiles[i + 1]) / 2);
    rawDensity.push(stepProb / dv);
  }

  // Interpolate onto an evenly-spaced value grid
  const grid = new Array(bins).fill(0);
  for (let b = 0; b < bins; b++) {
    const x = lo + (span * (b + 0.5)) / bins;
    // find bracketing midX pair
    let j = 0;
    while (j < midX.length - 1 && midX[j + 1] < x) j++;
    if (x <= midX[0]) grid[b] = rawDensity[0];
    else if (x >= midX[midX.length - 1]) grid[b] = rawDensity[rawDensity.length - 1];
    else {
      const x0 = midX[j], x1 = midX[j + 1];
      const t = (x - x0) / Math.max(x1 - x0, 1e-9);
      grid[b] = rawDensity[j] * (1 - t) + rawDensity[j + 1] * t;
    }
  }

  // Wide smoothing kernel (9-point weighted) to remove residual noise
  const kernel = [1, 2, 3, 4, 5, 4, 3, 2, 1];
  const kSum = kernel.reduce((a, b) => a + b, 0);
  const smooth = grid.map((_, i) => {
    let sum = 0;
    for (let k = 0; k < kernel.length; k++) {
      const idx = Math.min(grid.length - 1, Math.max(0, i + k - 4));
      sum += grid[idx] * kernel[k];
    }
    return sum / kSum;
  });

  // Taper the outer few bins toward zero smoothly, so the curve settles
  // rather than ending on a noisy edge bump
  const taper = Math.max(3, Math.round(bins * 0.04));
  for (let i = 0; i < taper; i++) {
    const f = (i + 1) / (taper + 1);
    smooth[i] *= f;
    smooth[smooth.length - 1 - i] *= f;
  }

  const peak = Math.max(...smooth, 1e-9);
  return { lo, hi, span, values: smooth.map((v) => v / peak) };
}

function curvePath(d, close) {
  const step = W / (d.values.length - 1);
  const y = (v) => PAD + (1 - v) * (H - 2 * PAD - 24);
  let path = `M 0 ${y(d.values[0]).toFixed(1)}`;

  for (let i = 1; i < d.values.length; i++) {
    const x0 = (i - 1) * step;
    const x1 = i * step;
    const mid = (x0 + x1) / 2;
    path += ` C ${mid.toFixed(1)} ${y(d.values[i - 1]).toFixed(1)},` +
            ` ${mid.toFixed(1)} ${y(d.values[i]).toFixed(1)},` +
            ` ${x1.toFixed(1)} ${y(d.values[i]).toFixed(1)}`;
  }

  if (close) path += ` L ${W} ${H - 24} L 0 ${H - 24} Z`;
  return path;
}

function drawDistribution(p) {
  const pct = p.margin_percentiles;
  if (!pct || pct.length < 8) return;

  const d = density(pct);
  const xOf = (v) => ((v - d.lo) / d.span) * W;
  const clamp = (x) => Math.max(0, Math.min(W, x));

  $("dist-fill").setAttribute("d", curvePath(d, true));
  $("dist-line").setAttribute("d", curvePath(d, false));

  const band = $("dist-band");
  band.innerHTML = "";
  const rect = (x1, x2, cls) => {
    const r = document.createElementNS("http://www.w3.org/2000/svg", "rect");
    r.setAttribute("x", clamp(x1));
    r.setAttribute("y", 4);
    r.setAttribute("width", Math.max(0, clamp(x2) - clamp(x1)));
    r.setAttribute("height", H - 28);
    r.setAttribute("class", cls);
    band.appendChild(r);
  };
  rect(xOf(p.interval_90[0]), xOf(p.interval_90[1]), "band-rect-90");
  rect(xOf(p.interval_50[0]), xOf(p.interval_50[1]), "band-rect-50");

  const zero = $("dist-zero");
  const zeroX = clamp(xOf(0));
  zero.setAttribute("x1", zeroX);
  zero.setAttribute("x2", zeroX);
  zero.style.display = (0 >= d.lo && 0 <= d.hi) ? "" : "none";

  const med = $("dist-median");
  const medX = clamp(xOf(p.median_margin));
  med.setAttribute("x1", medX);
  med.setAttribute("x2", medX);

  $("dist-axis").innerHTML =
    `<span>${signed(d.lo)}</span>` +
    `<span>${p.median_margin >= 0 ? MARKEY : MOULTON} margin</span>` +
    `<span>${signed(d.hi)}</span>`;
}


// ---------------------------------------------------------------------------
// Maps
//
// Same mechanism as MI: real GeoJSON in lat/lon (CRS84) with FIPS codes,
// projected in-browser with an Albers conic tuned to Wisconsin (not
// Michigan's parallels -- WI runs roughly 42.5N-47.1N, -92.9 to -86.8).
//
// Two maps: COUNTED (margin in votes already reported) and STILL OUT (margin
// the model projects for each county's remainder). Unlike MI, there's no
// vote-mode reason for these to diverge sharply within a county -- Wisconsin
// counties report in random order, not early-then-Election-Day -- so the two
// maps mostly track each other, diverging when a county's OWN reported
// margin differs from its baseline (the credibility blend) rather than from
// a known within-county mode shift.
// ---------------------------------------------------------------------------

const MAP_W = 700;
const MAP_H = 430; // Massachusetts is wide and short (aspect ~1.63), unlike WI's portrait shape
const MAP_SCALE = 30;

const MAPS = [
  { id: "counted", label: "Counted so far",
    note: "Margin in the votes already reported. Grey has not reported." },
  { id: "remaining", label: "Still out",
    note: "Margin projected for the vote not yet counted. Grey is finished." },
];

let GEO = null;
let PATHS = {};
let LAST_COUNTIES = [];
let PINNED = null;
let VIEW = { x: 0, y: 0, k: 1 };

// --- projection: Albers conic tuned to Wisconsin ----------------------------

function albers() {
  const LAT0 = 42.06, LON0 = -71.72, P1 = 41.51, P2 = 42.61; // Massachusetts (41.24N-42.89N, -73.51 to -69.93)
  const rad = Math.PI / 180;
  const n = 0.5 * (Math.sin(P1 * rad) + Math.sin(P2 * rad));
  const C = Math.cos(P1 * rad) ** 2 + 2 * n * Math.sin(P1 * rad);
  const rho0 = Math.sqrt(C - 2 * n * Math.sin(LAT0 * rad)) / n;
  return ([lon, lat]) => {
    const theta = n * (lon - LON0) * rad;
    const rho = Math.sqrt(Math.max(C - 2 * n * Math.sin(lat * rad), 1e-12)) / n;
    return [rho * Math.sin(theta), rho0 - rho * Math.cos(theta)];
  };
}

function buildPaths(geo) {
  const project = albers();
  const projected = geo.features.map((f) => {
    // GeoJSON Polygon vs MultiPolygon have different nesting: Polygon's
    // coordinates are directly [ring, ring, ...], MultiPolygon's are
    // [[ring, ring, ...], [ring, ...], ...] (one extra level, one per
    // disconnected landmass). Wisconsin's county file is a mix -- 69 plain
    // Polygon counties, 3 MultiPolygon (island counties like Bayfield with
    // the Apostle Islands) -- so both must be normalized to the same shape
    // before rendering, or the first Polygon county throws and the whole
    // map silently disappears.
    const polys = f.geometry.type === "Polygon"
      ? [f.geometry.coordinates]
      : f.geometry.coordinates;
    return {
      name: f.properties.name,
      polys: polys.map((poly) => poly.map((r) => r.map(project))),
    };
  });

  let minx = Infinity, maxx = -Infinity, miny = Infinity, maxy = -Infinity;
  projected.forEach((f) => f.polys.forEach((poly) => poly.forEach((r) => r.forEach(([x, y]) => {
    if (x < minx) minx = x;
    if (x > maxx) maxx = x;
    if (y < miny) miny = y;
    if (y > maxy) maxy = y;
  }))));

  const pad = 8;
  const k = Math.min((MAP_W - 2 * pad) / (maxx - minx), (MAP_H - 2 * pad) / (maxy - miny));
  const ox = (MAP_W - (maxx - minx) * k) / 2;
  const oy = (MAP_H - (maxy - miny) * k) / 2;

  const out = {};
  projected.forEach((f) => {
    let d = "";
    f.polys.forEach((poly) => poly.forEach((r) => {
      if (r.length < 4) return;
      d += "M" + r.map(([x, y]) =>
        ((x - minx) * k + ox).toFixed(1) + " " + ((maxy - y) * k + oy).toFixed(1)
      ).join("L") + "Z";
    }));
    out[f.name] = d;
  });
  return out;
}

// --- color --------------------------------------------------------------

function rampColor(margin) {
  if (margin === null || margin === undefined) return "#C7CCC2";
  const t = Math.max(-1, Math.min(1, margin / MAP_SCALE));
  const mid = [232, 234, 227];
  const end = t >= 0 ? [31, 111, 107] : [150, 112, 26];
  const k = Math.pow(Math.abs(t), 0.75);
  const c = mid.map((m, i) => Math.round(m + (end[i] - m) * k));
  return `rgb(${c[0]},${c[1]},${c[2]})`;
}

function countyRow(name) {
  return LAST_COUNTIES.find((r) => r.town === name);
}

function mapValue(row, mode) {
  if (!row) return null;
  if (mode === "counted") return row.reporting ? row.margin : null;
  return row.remaining > 0 ? row.remainder_margin : null;
}

// --- build ----------------------------------------------------------------

async function loadGeo() {
  try {
    const res = await fetch("ma-towns.geojson", { cache: "no-store" });
    GEO = await res.json();
    PATHS = buildPaths(GEO);
    MAPS.forEach(buildOne);
    paintMaps();
  } catch (err) {
    const el = document.querySelector(".maps");
    if (el) el.hidden = true;
  }
}

function buildOne(map) {
  const g = document.getElementById("shapes-" + map.id);
  g.innerHTML = "";
  Object.entries(PATHS).forEach(([county, d]) => {
    const p = document.createElementNS("http://www.w3.org/2000/svg", "path");
    p.setAttribute("d", d);
    p.setAttribute("fill", "#C7CCC2");
    p.dataset.county = county;
    p.addEventListener("pointerenter", () => hover(county, map.id));
    p.addEventListener("pointerleave", () => hover(null, map.id));
    p.addEventListener("click", (e) => {
      e.stopPropagation();
      PINNED = PINNED === county ? null : county;
      paintDetail(PINNED || county);
      markSelected();
    });
    g.appendChild(p);
  });
  document.getElementById("note-" + map.id).textContent = map.note;
}

function paintMaps() {
  if (!GEO) return;
  MAPS.forEach((map) => {
    document.getElementById("shapes-" + map.id)
      .querySelectorAll("path").forEach((p) => {
        p.setAttribute("fill", rampColor(mapValue(countyRow(p.dataset.county), map.id)));
      });
    const empty = map.id === "counted" ? "no results yet" : "counting complete";
    document.getElementById("legend-" + map.id).innerHTML =
      `<div class="legend-scale">${[-1, -0.6, -0.3, 0, 0.3, 0.6, 1]
        .map((t) => `<span style="background:${rampColor(t * MAP_SCALE)}"></span>`).join("")}</div>
       <div class="legend-ends"><span>Moulton +${MAP_SCALE}</span><span>tie</span><span>Markey +${MAP_SCALE}</span></div>
       <div class="legend-none"><i></i>${empty}</div>`;
  });
  markSelected();
  paintDetail(PINNED || $("map-detail").dataset.county || null);
}

function markSelected() {
  MAPS.forEach((map) => {
    document.getElementById("shapes-" + map.id)
      .querySelectorAll("path").forEach((p) => {
        p.classList.toggle("sel", p.dataset.county === PINNED);
        p.classList.toggle("hov", p.dataset.county === HOVER);
      });
  });
  $("pin-note").hidden = !PINNED;
}

let HOVER = null;

function hover(county, _from) {
  HOVER = county;
  markSelected();
  if (county) paintDetail(county);
  else if (PINNED) paintDetail(PINNED);
}

// --- detail panel -----------------------------------------------------------

function paintDetail(county) {
  const box = $("map-detail");
  const r = county && countyRow(county);
  if (!r) {
    box.innerHTML = `<p class="map-hint">Hover a county on either map. Click to pin it.</p>`;
    box.dataset.county = "";
    return;
  }
  box.dataset.county = county;
  const cls = (v) => (v >= 0 ? "v-markey" : "v-moulton");

  box.innerHTML = `
    <h3>${county}</h3>
    <dl>
      <dt>Region</dt><dd>${r.region}</dd>
      <dt>Counted</dt><dd>${num.format(r.votes)} of ${num.format(r.projected_total)}</dd>
      <dt>Margin so far</dt>
      <dd class="${r.margin === null ? "" : cls(r.margin)}">${r.margin === null ? "—" : signed(r.margin)}</dd>
      <dt>Pre-election baseline</dt><dd>${signed(r.expected_baseline)}</dd>
      <dt>Town-level swing</dt><dd>${signed(r.town_shift)}</dd>
      <dt>Still out</dt><dd>${num.format(r.remaining)}</dd>
      <dt>Remainder projects</dt>
      <dd class="${cls(r.remainder_margin)}">${signed(r.remainder_margin)}</dd>
    </dl>
    <p class="split-note">
      ${r.calibrated_turnout && r.calibrated_turnout !== r.projected_total
        ? `Turnout here is now projected at ${num.format(r.calibrated_turnout)}, revised from the pre-election baseline based on this county's own reporting pace.`
        : `Turnout still running on the pre-election baseline projection.`}
    </p>`;
}

// --- zoom and pan -----------------------------------------------------------

function applyView() {
  const t = `translate(${VIEW.x} ${VIEW.y}) scale(${VIEW.k})`;
  MAPS.forEach((m) => {
    document.getElementById("shapes-" + m.id).setAttribute("transform", t);
    document.getElementById("shapes-" + m.id)
      .style.setProperty("--stroke", (0.6 / VIEW.k).toFixed(2) + "px");
  });
  $("zoom-reset").hidden = VIEW.k === 1 && VIEW.x === 0 && VIEW.y === 0;
}

function zoomAt(svg, factor, cx, cy) {
  const box = svg.getBoundingClientRect();
  const px = ((cx - box.left) / box.width) * MAP_W;
  const py = ((cy - box.top) / box.height) * MAP_H;
  const k = Math.max(1, Math.min(12, VIEW.k * factor));
  VIEW.x = px - ((px - VIEW.x) / VIEW.k) * k;
  VIEW.y = py - ((py - VIEW.y) / VIEW.k) * k;
  VIEW.k = k;
  clampView();
  applyView();
}

function clampView() {
  const limX = MAP_W * (VIEW.k - 1);
  const limY = MAP_H * (VIEW.k - 1);
  VIEW.x = Math.max(-limX, Math.min(0, VIEW.x));
  VIEW.y = Math.max(-limY, Math.min(0, VIEW.y));
}

function initMapInteraction() {
  MAPS.forEach((map) => {
    const svg = document.getElementById("svg-" + map.id);

    svg.addEventListener("wheel", (e) => {
      e.preventDefault();
      zoomAt(svg, e.deltaY < 0 ? 1.18 : 1 / 1.18, e.clientX, e.clientY);
    }, { passive: false });

    let drag = null;
    svg.addEventListener("pointerdown", (e) => {
      if (VIEW.k === 1) return;
      drag = { x: e.clientX, y: e.clientY, vx: VIEW.x, vy: VIEW.y };
      svg.setPointerCapture(e.pointerId);
      svg.classList.add("dragging");
    });
    svg.addEventListener("pointermove", (e) => {
      if (!drag) return;
      const box = svg.getBoundingClientRect();
      VIEW.x = drag.vx + ((e.clientX - drag.x) / box.width) * MAP_W;
      VIEW.y = drag.vy + ((e.clientY - drag.y) / box.height) * MAP_H;
      clampView();
      applyView();
    });
    const stop = (e) => {
      if (!drag) return;
      drag = null;
      svg.releasePointerCapture(e.pointerId);
      svg.classList.remove("dragging");
    };
    svg.addEventListener("pointerup", stop);
    svg.addEventListener("pointercancel", stop);
  });

  $("zoom-in").addEventListener("click", () => {
    const svg = document.getElementById("svg-counted");
    const b = svg.getBoundingClientRect();
    zoomAt(svg, 1.5, b.left + b.width / 2, b.top + b.height / 2);
  });
  $("zoom-out").addEventListener("click", () => {
    const svg = document.getElementById("svg-counted");
    const b = svg.getBoundingClientRect();
    zoomAt(svg, 1 / 1.5, b.left + b.width / 2, b.top + b.height / 2);
  });
  $("zoom-reset").addEventListener("click", () => {
    VIEW = { x: 0, y: 0, k: 1 };
    applyView();
  });
  $("unpin").addEventListener("click", () => {
    PINNED = null;
    markSelected();
    paintDetail(null);
  });
}

// ---------------------------------------------------------------------------

let lastMargin = null;

function animateMargin(el, to) {
  const from = lastMargin;
  lastMargin = to;
  if (reduceMotion || from === null || Math.abs(to - from) < 0.05) {
    el.textContent = signed(to);
    return;
  }
  const start = performance.now();
  const dur = 550;
  const tick = (now) => {
    const t = Math.min(1, (now - start) / dur);
    const eased = 1 - Math.pow(1 - t, 3);
    el.textContent = signed(from + (to - from) * eased);
    if (t < 1) requestAnimationFrame(tick);
  };
  requestAnimationFrame(tick);
}

function renderCounties(all) {
  const rows = (all || []).filter((r) => r.reporting);
  const body = $("county-rows");
  if (!rows.length) {
    body.innerHTML = `<tr class="empty"><td colspan="7">No towns reporting yet.</td></tr>`;
    return;
  }
  body.innerHTML = rows.map((r) => {
    const cls = r.margin >= 0 ? "v-markey" : "v-moulton";
    const sw = r.vs_expected >= 0 ? "v-markey" : "v-moulton";
    return `<tr>
      <td class="name">${r.town}</td>
      <td class="num">${r.pct_of_projected.toFixed(0)}%</td>
      <td class="num">${num.format(r.markey)}</td>
      <td class="num">${num.format(r.moulton)}</td>
      <td class="num ${cls}">${signed(r.margin)}</td>
      <td class="num">${signed(r.expected_baseline)}</td>
      <td class="num ${sw}">${signed(r.vs_expected)}</td>
    </tr>`;
  }).join("");
}

function renderRegions(shifts) {
  const wrap = $("regions");
  const entries = Object.entries(shifts).sort((a, b) => b[1] - a[1]);
  const max = Math.max(2, ...entries.map(([, v]) => Math.abs(v)));

  wrap.innerHTML = entries.map(([name, v]) => {
    const half = (Math.abs(v) / max) * 50;
    const pos = v >= 0;
    return `<div class="region">
      <div>
        <div class="region-name">${name}</div>
        <div class="region-track">
          <span class="region-mid"></span>
          <span class="region-fill ${pos ? "pos" : "neg"}"
                style="left:${pos ? 50 : 50 - half}%;width:${half}%"></span>
        </div>
      </div>
      <div class="region-val ${pos ? "v-markey" : "v-moulton"}">${signed(v, 2)}</div>
    </div>`;
  }).join("");
}

function renderShareRanges(shareRanges) {
  const grid = $("range-grid");
  if (!shareRanges) { grid.innerHTML = ""; return; }
  const order = [["markey", "Markey"], ["moulton", "Moulton"]];
  grid.innerHTML = order.map(([key, label]) => {
    const r = shareRanges[key];
    if (!r) return "";
    return `<div class="range-cell">
      <div class="range-name ${key}">${label}</div>
      <div class="range-median ${key}">${r.median.toFixed(1)}%</div>
      <div class="range-bands">
        50%: ${r.range_50[0].toFixed(1)}–${r.range_50[1].toFixed(1)}<br>
        90%: ${r.range_90[0].toFixed(1)}–${r.range_90[1].toFixed(1)}
      </div>
    </div>`;
  }).join("");
}

function render(data) {
  const p = data.projection;
  const c = data.counted;
  const d = data.diagnostics;
  const t = data.turnout;

  const leadMarkey = p.median_margin >= 0;
  const leader = leadMarkey ? MARKEY : MOULTON;
  const prob = leadMarkey ? p.markey_win_probability : p.moulton_win_probability;

  $("lead-name").textContent = leader;
  $("lead-name").className = "verdict-name " + (leadMarkey ? "markey" : "moulton");
  $("lead-margin").className = "verdict-number " + (leadMarkey ? "markey" : "moulton");
  animateMargin($("lead-margin"), Math.abs(p.median_margin));

  $("verdict-sub").textContent = d.towns_reporting === 0
    ? "Pre-election baseline. No towns reporting."
    : `From ${d.towns_reporting} ${d.towns_reporting === 1 ? "town" : "towns"}` +
      ` and ${c.pct_of_projected_turnout.toFixed(1)}% of the projected vote.`;

  drawDistribution(p);

  $("range-90").textContent = `${signed(p.interval_90[0])} to ${signed(p.interval_90[1])}`;
  $("range-50").textContent = `${signed(p.interval_50[0])} to ${signed(p.interval_50[1])}`;

  $("win-prob").textContent = (prob * 100).toFixed(prob > 0.995 ? 1 : 0) + "%";
  $("win-note").textContent = `${leader} wins in ${(prob * 100).toFixed(1)}% of simulations`;

  $("counted").textContent = c.pct_of_projected_turnout.toFixed(1) + "%";
  $("precincts").textContent =
    c.pct_precincts_reporting == null ? "—" : c.pct_precincts_reporting + "%";

  $("turnout").textContent = t.projected ? num.format(t.projected) : "—";

  renderShareRanges(p.share_ranges);

  const markeyShare = p.markey_pct;
  const moultonShare = p.moulton_pct;
  $("markey-votes").textContent = num.format(c.markey || 0);
  $("moulton-votes").textContent = num.format(c.moulton || 0);
  $("tally-markey").style.width = markeyShare + "%";
  $("tally-moulton").style.width = moultonShare + "%";
  $("markey-pct").textContent = markeyShare.toFixed(1) + "%";
  $("moulton-pct").textContent = moultonShare.toFixed(1) + "%";

  LAST_COUNTIES = data.towns || [];
  renderCounties(data.towns);
  paintMaps();
  renderRegions(data.regional_shift || {});

  $("d-counties").textContent = d.towns_reporting;
  $("d-shift").textContent = signed(d.statewide_shift, 2);
  $("d-ci50").textContent = `${signed(p.interval_50[0])} to ${signed(p.interval_50[1])}`;
  $("d-ci90").textContent = `${signed(p.interval_90[0])} to ${signed(p.interval_90[1])}`;

  const stamp = new Date(data.updated_at);
  const stale = Date.now() - stamp.getTime() > STALE_AFTER_MS;
  setPulse(stale ? "stale" : "live", stale ? "feed stale" : "live", stamp);

  if (d.unmatched_towns && d.unmatched_towns.length) {
    $("alert").textContent =
      "Not matched to a model town, and excluded from the projection: " +
      d.unmatched_towns.join(", ");
    $("alert").hidden = false;
  } else {
    $("alert").hidden = true;
  }
}

function setPulse(state, label, stamp) {
  $("pulse").dataset.state = state;
  $("pulse-label").textContent = label;
  if (stamp) {
    $("stamp").textContent = stamp.toLocaleTimeString([], {
      hour: "2-digit", minute: "2-digit", second: "2-digit",
    });
  }
}

async function tick() {
  try {
    const res = await fetch(API_BASE + "/api/projection", { cache: "no-store" });
    if (res.status === 503) {
      setPulse("connecting", "waiting for first results");
      return;
    }
    if (!res.ok) throw new Error("HTTP " + res.status);
    render(await res.json());
  } catch (err) {
    setPulse("stale", "reconnecting");
  }
}

initMapInteraction();
loadGeo();
tick();
setInterval(tick, REFRESH_MS);
