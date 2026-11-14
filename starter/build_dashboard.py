#!/usr/bin/env python3
"""
Generates a 3-Tab Interactive Presentation & Operator Decision Dashboard
(`operator_dashboard.html`) from real telemetry, SDA feeds, and submission.json.
"""
import json
from pathlib import Path
import pandas as pd


def build_dashboard(data_dir: Path, sub_path: Path, out_html: Path):
    with open(sub_path) as f:
        sub_data = json.load(f)
    detections = sub_data["detections"]

    tele = pd.read_csv(data_dir / "farm/tractor_telemetry.csv.gz", parse_dates=["ts"])
    tele["ts"] = pd.to_datetime(tele["ts"], utc=True)

    with open(data_dir / "sda/scintillation_series.json") as f:
        s4 = pd.DataFrame(json.load(f)["series"])
    s4["ts"] = pd.to_datetime(s4["ts"], utc=True)

    with open(data_dir / "sda/sgi_indices.json") as f:
        sgi = json.load(f)
    kp = pd.DataFrame(sgi["kp"])
    kp["ts"] = pd.to_datetime(kp["ts"], utc=True)

    eval_start = pd.Timestamp("2026-05-06T00:00:00Z")
    eval_end = pd.Timestamp("2026-05-07T00:00:00Z")
    sub_tele = tele[(tele.ts >= eval_start) & (tele.ts < eval_end)].copy()
    sub_tele["abs_xte"] = sub_tele["cross_track_error_m"].abs()
    tractors = sorted(sub_tele["tractor_id"].dropna().unique().tolist())
    total_samples = len(sub_tele)

    grid = (
        sub_tele.set_index("ts")
        .groupby("tractor_id")
        .resample("5min", include_groups=False)
        .agg(
            cn0=("mean_cn0_dbhz", "mean"),
            xte=("abs_xte", "max"),
            speed=("speed_mps", "mean"),
            link=("base_link_ok", "mean"),
        )
        .reset_index()
    )
    grid = pd.merge_asof(grid.sort_values("ts"), s4.sort_values("ts"), on="ts")
    grid = pd.merge_asof(grid.sort_values("ts"), kp.sort_values("ts"), on="ts")

    timestamps = sorted(grid["ts"].unique())
    ts_labels = [pd.Timestamp(t).strftime("%H:%M") for t in timestamps]

    series = {}
    for tr in tractors:
        tr_df = grid[grid["tractor_id"] == tr].set_index("ts").reindex(timestamps)
        series[tr] = {
            "cn0": [round(float(v), 2) if pd.notna(v) else None for v in tr_df["cn0"]],
            "xte": [round(float(v), 3) if pd.notna(v) else None for v in tr_df["xte"]],
            "link": [round(float(v), 2) if pd.notna(v) else None for v in tr_df["link"]],
        }

    s4_df = grid[grid["tractor_id"] == tractors[0]].set_index("ts").reindex(timestamps)
    sda_series = {
        "s4": [round(float(v), 3) if pd.notna(v) else 0.0 for v in s4_df["s4"]],
        "kp": [round(float(v), 1) if pd.notna(v) else 1.0 for v in s4_df.get("kp", s4_df.iloc[:, -1])],
    }

    spatial = {}
    for tr in tractors:
        tr_pts = sub_tele[(sub_tele["tractor_id"] == tr) & (sub_tele["speed_mps"] > 0)].iloc[::30]
        spatial[tr] = [
            {"lat": round(float(r.lat), 6), "lon": round(float(r.lon), 6)}
            for r in tr_pts.itertuples()
        ]

    macro_dets = [d for d in detections if d["class"] != "MULTIPATH"]
    mp_dets = [d for d in detections if d["class"] == "MULTIPATH"]

    payload = {
        "tractors": tractors,
        "labels": ts_labels,
        "series": series,
        "sda": sda_series,
        "macro_detections": macro_dets,
        "multipath_summary": {
            "count": len(mp_dets),
            "sample": mp_dets[0] if mp_dets else None,
            "by_tractor": {tr: sum(1 for m in mp_dets if tr in m["tractors"]) for tr in tractors},
        },
        "spatial": spatial,
    }

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>Straight Rows — Autonomous Fleet GNSS Diagnostics</title>
  <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
  <style>
    :root {{
      --bg: #090e1a; --panel: #111827; --border: #1f2937; --text: #f3f4f6; --muted: #9ca3af;
      --sw: #f59e0b; --jam: #ef4444; --spf: #a855f7; --eq: #3b82f6; --mp: #10b981; --accent: #38bdf8;
    }}
    * {{ box-sizing: border-box; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; }}
    body {{ margin: 0; background: var(--bg); color: var(--text); }}
    header {{
      background: linear-gradient(90deg, #0f172a 0%, #1e1b4b 100%);
      padding: 16px 28px; border-bottom: 1px solid var(--border);
      display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 16px;
    }}
    header h1 {{ margin: 0; font-size: 1.32rem; letter-spacing: 0.2px; }}
    header p {{ margin: 4px 0 0; color: var(--muted); font-size: 0.85rem; }}

    .header-right {{ display: flex; align-items: center; gap: 20px; flex-wrap: wrap; }}
    .nav-tabs {{ display: flex; gap: 6px; background: #090e1a; padding: 5px; border-radius: 10px; border: 1px solid var(--border); }}
    .tab-btn {{
      background: transparent; color: var(--muted); border: none; padding: 7px 12px;
      border-radius: 7px; font-weight: 600; font-size: 0.82rem; cursor: pointer; transition: all 0.15s;
    }}
    .tab-btn.active {{ background: #1e293b; color: var(--accent); box-shadow: 0 1px 3px rgba(0,0,0,0.4); }}

    .kpi-row {{ display: flex; gap: 10px; flex-wrap: wrap; }}
    .kpi {{
      background: rgba(255,255,255,0.04); border: 1px solid rgba(255,255,255,0.09);
      padding: 6px 11px; border-radius: 8px; text-align: right;
    }}
    .kpi .val {{ font-size: 1.05rem; font-weight: 700; color: var(--accent); }}
    .kpi .lbl {{ font-size: 0.68rem; color: var(--muted); text-transform: uppercase; }}

    .page {{ display: none; max-width: 1600px; margin: 0 auto; padding: 18px 24px; }}
    .page.active {{ display: block; }}
    .grid-main {{ display: grid; grid-template-columns: 360px 1fr; gap: 18px; }}
    @media (max-width: 1050px) {{ .grid-main {{ grid-template-columns: 1fr; }} }}
    .card {{ background: var(--panel); border: 1px solid var(--border); border-radius: 12px; padding: 16px; }}
    .card h2 {{ margin: 0 0 12px; font-size: 0.98rem; color: #e5e7eb; display: flex; justify-content: space-between; align-items: center; }}
    .ep-list {{ display: flex; flex-direction: column; gap: 9px; }}
    .ep-item {{
      background: #0f172a; border: 1px solid #1e293b; border-left: 5px solid #64748b;
      border-radius: 8px; padding: 10px 12px; cursor: pointer; transition: all 0.15s ease;
    }}
    .ep-item:hover, .ep-item.active {{ background: #1e293b; border-color: var(--accent); transform: translateX(2px); }}
    .ep-top {{ display: flex; justify-content: space-between; align-items: center; font-weight: 700; font-size: 0.88rem; }}
    .ep-time {{ font-size: 0.76rem; color: var(--muted); margin-top: 4px; font-family: monospace; }}
    .badge {{ padding: 2px 8px; border-radius: 999px; font-size: 0.7rem; font-weight: 700; color: #fff; }}
    .c-SPACE_WEATHER {{ border-left-color: var(--sw); }} .b-SPACE_WEATHER {{ background: var(--sw); color: #000; }}
    .c-JAMMING {{ border-left-color: var(--jam); }} .b-JAMMING {{ background: var(--jam); }}
    .c-SPOOFING {{ border-left-color: var(--spf); }} .b-SPOOFING {{ background: var(--spf); }}
    .c-EQUIPMENT_FAULT {{ border-left-color: var(--eq); }} .b-EQUIPMENT_FAULT {{ background: var(--eq); }}
    .c-MULTIPATH {{ border-left-color: var(--mp); }} .b-MULTIPATH {{ background: var(--mp); color: #000; }}
    .forensic-box {{ background: #0f172a; border: 1px solid #1e293b; border-radius: 10px; padding: 15px; }}
    .action-banner {{
      display: inline-block; padding: 6px 12px; border-radius: 6px; font-weight: 800;
      font-size: 0.84rem; letter-spacing: 0.4px; background: rgba(56, 189, 248, 0.15); color: var(--accent);
      border: 1px solid rgba(56, 189, 248, 0.4);
    }}
    .conf-meter {{ display: flex; align-items: center; gap: 10px; margin-top: 10px; background: #090e1a; padding: 8px 12px; border-radius: 8px; border: 1px solid #1f2937; }}
    .conf-track {{ flex: 1; height: 8px; background: #1f2937; border-radius: 999px; overflow: hidden; }}
    .conf-fill {{ height: 100%; background: linear-gradient(90deg, #38bdf8, #10b981); border-radius: 999px; }}
    .ev-grid {{ display: grid; grid-template-columns: repeat(3, 1fr); gap: 12px; margin-top: 12px; }}
    @media (max-width: 1100px) {{ .ev-grid {{ grid-template-columns: 1fr; }} }}
    .ev-card {{ background: #090e1a; border: 1px solid #1f2937; border-top: 3px solid #38bdf8; border-radius: 8px; padding: 11px 13px; }}
    .ev-card.sda {{ border-top-color: #f59e0b; }}
    .ev-card.ruleout {{ border-top-color: #10b981; }}
    .ev-title {{ font-size: 0.72rem; font-weight: 800; text-transform: uppercase; letter-spacing: 0.5px; color: var(--muted); margin-bottom: 6px; }}
    .ev-body {{ font-size: 0.86rem; line-height: 1.48; color: #e5e7eb; }}
    .charts-grid {{ display: grid; grid-template-columns: 1fr; gap: 10px; }}
    .chart-wrap {{ position: relative; height: 165px; width: 100%; }}
    .bottom-grid {{ display: grid; grid-template-columns: 1.2fr 0.8fr; gap: 16px; margin-top: 16px; }}
    @media (max-width: 900px) {{ .bottom-grid {{ grid-template-columns: 1fr; }} }}
    canvas#fieldCanvas {{ width: 100%; height: 215px; background: #070b14; border-radius: 8px; border: 1px solid #1f2937; }}
    .legend-pill {{ font-size: 0.75rem; color: var(--muted); }}

    /* PREVENT HORIZONTAL SCROLL ON CODE BLOCKS */
    pre.code-block {{
      background: #060a12; border: 1px solid #1e293b; border-radius: 8px; padding: 14px;
      color: #a5f3fc; font-family: "Cascadia Code", Consolas, monospace; font-size: 0.82rem;
      white-space: pre-wrap; word-break: break-all; overflow-x: hidden; line-height: 1.45; margin: 10px 0 0;
    }}
    .ch-grid {{ display: grid; grid-template-columns: 280px 1fr; gap: 18px; }}
    @media (max-width: 900px) {{ .ch-grid {{ grid-template-columns: 1fr; }} }}
    table.matrix {{ width: 100%; border-collapse: collapse; font-size: 0.83rem; }}
    table.matrix th, table.matrix td {{ padding: 8px 10px; border-bottom: 1px solid #1f2937; text-align: left; }}
    table.matrix th {{ color: var(--muted); font-size: 0.72rem; text-transform: uppercase; }}
    .tab3-grid {{ display: grid; grid-template-columns: repeat(3, 1fr); gap: 16px; }}
    @media (max-width: 1100px) {{ .tab3-grid {{ grid-template-columns: 1fr; }} }}
  </style>
</head>
<body>
  <header>
    <div>
      <h1>Straight Rows — Autonomous Fleet GNSS Diagnostics</h1>
      <p>Real-Time Anomaly Attribution &amp; Operator Decision Support &bull; Fleet: {", ".join(tractors)} &bull; May 6 UTC</p>
    </div>
    <div class="header-right">
      <div class="kpi-row">
        <div class="kpi"><div class="val">{len(tractors)} Tractors</div><div class="lbl">Fleet</div></div>
        <div class="kpi"><div class="val">{len(macro_dets)} Alerts</div><div class="lbl">Incidents</div></div>
        <div class="kpi"><div class="val" style="color:var(--mp);">{len(mp_dets)} Cleared</div><div class="lbl">Multipath</div></div>
      </div>
      <div class="nav-tabs">
        <button class="tab-btn active" onclick="switchTab('mission', this)">1. Mission Control</button>
        <button class="tab-btn" onclick="switchTab('code', this)">2. Code &amp; Engine</button>
        <button class="tab-btn" onclick="switchTab('bench', this)">3. Generalizability &amp; Roadmap</button>
      </div>
    </div>
  </header>

  <!-- TAB 1: LIVE MISSION CONTROL -->
  <div id="tab-mission" class="page active">
    <div class="grid-main">
      <div class="card">
        <h2>Detected Episodes <span class="legend-pill">Click to inspect</span></h2>
        <div class="ep-list" id="episodeList"></div>
      </div>

      <div>
        <div class="card" style="margin-bottom: 16px;">
          <h2>Recommended Operator Decision &amp; Evidence Breakdown</h2>
          <div class="forensic-box" id="forensicBox"></div>
        </div>

        <div class="card">
          <h2>Fleet Telemetry &amp; Environmental Context (Synchronized Timeline)</h2>
          <div class="charts-grid">
            <div class="chart-wrap"><canvas id="cn0Chart"></canvas></div>
            <div class="chart-wrap"><canvas id="xteChart"></canvas></div>
            <div class="chart-wrap" style="height: 130px;"><canvas id="envChart"></canvas></div>
          </div>

          <div class="bottom-grid">
            <div>
              <h2 style="margin-top:0; display:flex; justify-content:space-between; align-items:center;">
                <span>Field Spatial Tracks &amp; Multipath Cluster</span>
                <span class="legend-pill" style="color:#10b981; font-weight:700;">● Discovered Obstacle Zone</span>
              </h2>
              <canvas id="fieldCanvas" width="520" height="205"></canvas>
              <div class="legend-pill" style="margin-top:6px;">
                <strong>How to read:</strong> Colored dots are tractor GPS tracks across the field. The dashed green line highlights where 17-second signal dips repeat automatically within 45 meters (identifying a stationary tree line/barn headland).
              </div>
            </div>
            <div>
              <h2 style="margin-top:0;">Overlapping Event Isolation</h2>
              <div class="forensic-box" style="font-size:0.84rem; line-height:1.55; color:#d1d5db;">
                <strong>Multi-Label Architecture:</strong><br>
                At <strong>13:00Z–16:00Z</strong>, a regional solar storm (S4 &ge; 0.15) depresses the entire fleet. Instead of locking out other detectors, our 5 independent channels catch <strong>T1 Spoofing</strong> at <strong>16:31Z</strong> and brief <strong>1-Hz Multipath turns</strong> without false halts.
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  </div>

  <!-- TAB 2: INTERACTIVE CODE & 5-CHANNEL ENGINE -->
  <div id="tab-code" class="page">
    <div class="card" style="margin-bottom: 16px;">
      <h2>5-Class Physical Attribution &amp; Decision Matrix</h2>
      <div class="forensic-box" style="padding:6px 10px;">
        <table class="matrix">
          <thead>
            <tr><th>Cause Class</th><th>Primary Physical Invariant</th><th>Operator Decision</th></tr>
          </thead>
          <tbody>
            <tr><td><span class="badge b-SPACE_WEATHER">SPACE_WEATHER</span></td><td>Fleet-wide C/N0 drop + S4 &ge; 0.15, RTK link up</td><td><code>HALT_OR_DEGRADED_MODE</code></td></tr>
            <tr><td><span class="badge b-JAMMING">JAMMING</span></td><td>&ge;2 tractors C/N0 collapse, RTK link up</td><td><code>HALT_AND_REPORT</code></td></tr>
            <tr><td><span class="badge b-SPOOFING">SPOOFING</span></td><td>1 tractor XTE drift + healthy C/N0 (&gt;41 dB-Hz)</td><td><code>HALT_CROSSCHECK_AND_REPORT</code></td></tr>
            <tr><td><span class="badge b-EQUIPMENT_FAULT">EQUIPMENT_FAULT</span></td><td>1 tractor RTK link loss or isolated LNA fade</td><td><code>SERVICE_MACHINE_OTHERS_CONTINUE</code></td></tr>
            <tr><td><span class="badge b-MULTIPATH">MULTIPATH</span></td><td>&lt;40s 1-Hz C/N0 notch repeating within 45m</td><td><code>CONTINUE_FALSE_ALARM</code></td></tr>
          </tbody>
        </table>
      </div>
    </div>

    <div class="ch-grid">
      <div class="card">
        <h2>Attribution Channels</h2>
        <div class="ep-list" id="channelList"></div>
      </div>
      <div class="card" id="channelDetail"></div>
    </div>
  </div>

  <!-- TAB 3: GENERALIZABILITY, LIMITATIONS & ROADMAP -->
  <div id="tab-bench" class="page">
    <div class="tab3-grid">
      <div class="card">
        <h2>1. How the Code is Generic &amp; Scalable</h2>
        <div class="forensic-box" style="font-size:0.86rem; line-height:1.6; color:#d1d5db;">
          <strong>Zero Hardcoded Coordinates:</strong> Multipath is discovered dynamically using 45m geodesic spatial clustering rather than hardcoded farm boundaries.<br><br>
          <strong>Dynamic Fleet Sizing:</strong> Tractors are discovered automatically from telemetry headers (`tele['tractor_id'].unique()`), supporting any fleet size ($N \ge 2$).<br><br>
          <strong>Relative Fleet Consensus:</strong> Spoofing and equipment faults compare each machine against live fleet maximums and medians, adapting automatically to seasonal noise shifts.
        </div>
      </div>

      <div class="card">
        <h2>2. Current System Limitations</h2>
        <div class="forensic-box" style="font-size:0.86rem; line-height:1.6; color:#d1d5db;">
          <strong>Low-Power Single Jammer Ambiguity:</strong> A localized jammer affecting only one tractor during quiet space weather can occasionally resemble an antenna hardware fault.<br><br>
          <strong>Gradual Spoofing Onset Latency:</strong> Extremely slow position ramps (&lt;0.12 m/min) require several minutes of divergence before crossing the XTE detection threshold.
        </div>
      </div>

      <div class="card">
        <h2>3. Potential Future Upgrades</h2>
        <div class="forensic-box" style="font-size:0.86rem; line-height:1.6; color:#d1d5db;">
          <strong>Receiver AGC Gain Telemetry:</strong> Adding automatic gain control telemetry would instantly distinguish high-power RF jamming from natural ionospheric fading.<br><br>
          <strong>Dual-Antenna GNSS Heading:</strong> Incorporating dual-antenna heading vectors would provide instant proof of spoofing the moment GNSS heading diverges from wheel/IMU steering.
        </div>
      </div>
    </div>
  </div>

  <script>
    const DATA = {json.dumps(payload)};

    function switchTab(name, btn) {{
      document.querySelectorAll(".page").forEach(p => p.classList.remove("active"));
      document.querySelectorAll(".tab-btn").forEach(b => b.classList.remove("active"));
      document.getElementById("tab-" + name).classList.add("active");
      btn.classList.add("active");
    }}

    const epListEl = document.getElementById("episodeList");
    const forensicEl = document.getElementById("forensicBox");

    const displayEpisodes = [...DATA.macro_detections];
    if (DATA.multipath_summary.sample) {{
      displayEpisodes.push({{
        ...DATA.multipath_summary.sample,
        _is_mp_group: true,
        _mp_count: DATA.multipath_summary.count,
        _mp_by_tr: DATA.multipath_summary.by_tractor
      }});
    }}

    function cleanEvText(str) {{
      return str.replace(/^[^:]+:\\s*/, "");
    }}

    function renderForensic(ep, idx) {{
      document.querySelectorAll("#episodeList .ep-item").forEach((el, i) => el.classList.toggle("active", i === idx));
      const title = ep._is_mp_group
        ? `MULTIPATH (${{ep._mp_count}} Recurrent 1-Hz Headland Reflections)`
        : `${{ep.class}} — Tractors: ${{ep.tractors.join(", ")}}`;
      const trSummary = DATA.tractors.map(t => `${{t}}: ${{ep._mp_by_tr ? ep._mp_by_tr[t] : 0}} turns`).join(" | ");
      const timeStr = ep._is_mp_group
        ? `${{trSummary}} (all &le; 180s)`
        : `${{ep.start}} &rarr; ${{ep.end}}`;

      const pct = Math.round(ep.confidence * 100);
      const tier = pct >= 93 ? "HIGH CERTAINTY (CORROBORATED)" : "MODERATE-HIGH (LATENCY / MARGIN)";

      forensicEl.innerHTML = `
        <div style="display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; gap:8px;">
          <div style="font-size:1.05rem; font-weight:700;">${{title}}</div>
          <div class="action-banner">ACTION: ${{ep.action}}</div>
        </div>
        <div style="font-family:monospace; font-size:0.8rem; color:#9ca3af; margin-top:4px;">${{timeStr}}</div>
        <div class="conf-meter">
          <span style="font-size:0.76rem; font-weight:700; color:#9ca3af;">CONFIDENCE</span>
          <div class="conf-track"><div class="conf-fill" style="width:${{pct}}%;"></div></div>
          <span style="font-size:0.82rem; font-weight:800; color:#34d399;">${{pct}}% &bull; ${{tier}}</span>
        </div>
        <div class="ev-grid">
          <div class="ev-card">
            <div class="ev-title">1. Farm Telemetry Signal</div>
            <div class="ev-body">${{cleanEvText(ep.evidence[0] || "")}}</div>
          </div>
          <div class="ev-card sda">
            <div class="ev-title">2. SDA &amp; Spatial Context</div>
            <div class="ev-body">${{cleanEvText(ep.evidence[1] || "")}}</div>
          </div>
          <div class="ev-card ruleout">
            <div class="ev-title">3. Other Causes Ruled Out</div>
            <div class="ev-body">${{cleanEvText(ep.evidence[2] || "")}}</div>
          </div>
        </div>
      `;
    }}

    displayEpisodes.forEach((ep, idx) => {{
      const div = document.createElement("div");
      div.className = `ep-item c-${{ep.class}}`;
      const label = ep._is_mp_group ? `MULTIPATH (&times;${{ep._mp_count}})` : ep.class;
      const trText = ep._is_mp_group ? DATA.tractors.join(", ") : ep.tractors.join(", ");
      const tShort = ep._is_mp_group ? "Recurring 1-Hz Spatial Cluster Turns" : `${{ep.start.slice(11,16)}}Z &rarr; ${{ep.end.slice(11,16)}}Z`;
      div.innerHTML = `
        <div class="ep-top">
          <span><span class="badge b-${{ep.class}}">${{label}}</span></span>
          <span style="font-size:0.78rem; color:#93c5fd;">${{trText}}</span>
        </div>
        <div class="ep-time">${{tShort}} &bull; Conf: ${{(ep.confidence*100).toFixed(0)}}%</div>
      `;
      div.onclick = () => renderForensic(ep, idx);
      epListEl.appendChild(div);
    }});
    if (displayEpisodes.length > 0) renderForensic(displayEpisodes[0], 0);

    const palette = ["#38bdf8", "#34d399", "#fbbf24", "#f472b6", "#a78bfa"];
    const cn0Datasets = DATA.tractors.map((tr, i) => ({{
      label: `${{tr}} C/N0 (dB-Hz)`, data: DATA.series[tr].cn0, borderColor: palette[i % palette.length], borderWidth: 1.8, pointRadius: 0, yAxisID: "y"
    }}));

    new Chart(document.getElementById("cn0Chart"), {{
      type: "line",
      data: {{ labels: DATA.labels, datasets: cn0Datasets }},
      options: {{
        responsive: true, maintainAspectRatio: false,
        plugins: {{ legend: {{ labels: {{ color: "#d1d5db", boxWidth: 12, font: {{ size: 11 }} }} }} }},
        scales: {{
          x: {{ display: false }},
          y: {{ position: "left", title: {{ display: true, text: "C/N0 (dB-Hz)", color: "#9ca3af", font: {{ size: 10 }} }}, ticks: {{ color: "#9ca3af", font: {{ size: 10 }} }}, grid: {{ color: "#1f2937" }} }}
        }}
      }}
    }});

    const xteDatasets = DATA.tractors.map((tr, i) => ({{
      label: `${{tr}} Max XTE (m)`, data: DATA.series[tr].xte, borderColor: palette[i % palette.length], borderWidth: 1.8, pointRadius: 0
    }}));

    new Chart(document.getElementById("xteChart"), {{
      type: "line",
      data: {{ labels: DATA.labels, datasets: xteDatasets }},
      options: {{
        responsive: true, maintainAspectRatio: false,
        plugins: {{ legend: {{ labels: {{ color: "#d1d5db", boxWidth: 12, font: {{ size: 11 }} }} }} }},
        scales: {{
          x: {{ display: false }},
          y: {{ title: {{ display: true, text: "XTE Error (m)", color: "#9ca3af", font: {{ size: 10 }} }}, ticks: {{ color: "#9ca3af", font: {{ size: 10 }} }}, grid: {{ color: "#1f2937" }} }}
        }}
      }}
    }});

    new Chart(document.getElementById("envChart"), {{
      type: "line",
      data: {{
        labels: DATA.labels,
        datasets: [
          {{ label: "S4 Ionospheric Scintillation", data: DATA.sda.s4, borderColor: "#f87171", backgroundColor: "rgba(248,113,113,0.1)", fill: true, borderWidth: 1.5, pointRadius: 0, yAxisID: "y" }},
          {{ label: "Kp Geomagnetic Index", data: DATA.sda.kp, borderColor: "#fbbf24", borderWidth: 1.5, pointRadius: 0, yAxisID: "y1" }}
        ]
      }},
      options: {{
        responsive: true, maintainAspectRatio: false,
        plugins: {{ legend: {{ labels: {{ color: "#d1d5db", boxWidth: 12, font: {{ size: 11 }} }} }} }},
        scales: {{
          x: {{ ticks: {{ color: "#9ca3af", maxTicksLimit: 12, font: {{ size: 10 }} }}, grid: {{ color: "#1f2937" }} }},
          y: {{ position: "left", min: 0, max: 0.5, title: {{ display: true, text: "S4 Index", color: "#f87171", font: {{ size: 10 }} }}, ticks: {{ color: "#f87171", font: {{ size: 10 }} }}, grid: {{ color: "#1f2937" }} }},
          y1: {{ position: "right", min: 0, max: 9, title: {{ display: true, text: "Kp Index", color: "#fbbf24", font: {{ size: 10 }} }}, ticks: {{ color: "#fbbf24", font: {{ size: 10 }} }}, grid: {{ drawOnChartArea: false }} }}
        }}
      }}
    }});

    const canvas = document.getElementById("fieldCanvas");
    const ctx = canvas.getContext("2d");
    const allPts = DATA.tractors.flatMap(tr => DATA.spatial[tr]);
    if (allPts.length > 0) {{
      const minLat = Math.min(...allPts.map(p => p.lat)), maxLat = Math.max(...allPts.map(p => p.lat));
      const minLon = Math.min(...allPts.map(p => p.lon)), maxLon = Math.max(...allPts.map(p => p.lon));
      const mapX = lon => 28 + ((lon - minLon) / (maxLon - minLon || 1)) * (canvas.width - 56);
      const mapY = lat => canvas.height - 24 - ((lat - minLat) / (maxLat - minLat || 1)) * (canvas.height - 48);

      DATA.tractors.forEach((tr, i) => {{
        ctx.fillStyle = palette[i % palette.length];
        DATA.spatial[tr].forEach(p => ctx.fillRect(mapX(p.lon), mapY(p.lat), 2.5, 2.5));
      }});

      if (DATA.multipath_summary.sample && DATA.multipath_summary.sample.evidence[0]) {{
        const match = DATA.multipath_summary.sample.evidence[0].match(/near \\(([0-9.]+)/);
        const mpLat = match ? parseFloat(match[1]) : (minLat + (maxLat - minLat) * 0.72);
        const nhY = mapY(mpLat);
        ctx.strokeStyle = "#10b981";
        ctx.lineWidth = 2;
        ctx.setLineDash([6, 4]);
        ctx.beginPath(); ctx.moveTo(16, nhY); ctx.lineTo(canvas.width - 16, nhY); ctx.stroke();
        ctx.setLineDash([]);
        ctx.fillStyle = "#10b981";
        ctx.font = "bold 10px monospace";
        ctx.fillText("▲ DISCOVERED 45m OBSTACLE ZONE (MULTIPATH REFLECTION)", 22, Math.max(16, nhY - 6));
      }}
    }}

    const CHANNELS = [
      {{
        cls: "SPACE_WEATHER", title: "Channel 1: Space Weather",
        physics: "Regional ionospheric scintillation (S4 >= 0.15) depresses carrier-to-noise (C/N0) simultaneously across all moving tractors while RTK radio links stay intact.",
        why: "Uses regional S4 scintillation directly rather than global Kp alone, aligning carrier fading with local ionospheric turbulence.",
        code: `# Channel 1: SPACE_WEATHER — fleet-wide C/N0 drop coupled with elevated S4
sw_mask = all_moving & (s4_vec >= 0.15) & (fleet_cn0_mean < 39.0)
raw_episodes.extend(
    boolean_mask_to_intervals(ts_idx, sw_mask, "SPACE_WEATHER", tractors, max_gap_min=5, min_dur_min=15)
)`
      }},
      {{
        cls: "JAMMING", title: "Channel 2: RF Jamming",
        physics: "Localized RF interference collapses C/N0 and satellite tracking on >= 2 tractors simultaneously while RTK correction links remain up.",
        why: "Uses a dual threshold (clear-sky relative depression vs. storm-concurrent collapse) so terrestrial jamming is still caught even when overlaid on a solar storm.",
        code: `# Channel 2: JAMMING — multi-tractor C/N0 depression (quiet + storm dual gate)
clear_jam = (s4_vec < 0.15) & ((piv[("cn0", tr)] < 37.5) | ((piv[("cn0", tr)] < 40.2) & ((fleet_cn0_max - piv[("cn0", tr)]) > 2.0)))
storm_jam = (s4_vec >= 0.15) & (piv[("cn0", tr)] < 27.5) & (piv[("sats", tr)] < 8.8)
jam_per_tr[tr] = all_moving & (clear_jam | storm_jam)`
      }},
      {{
        cls: "SPOOFING", title: "Channel 3: GNSS Spoofing",
        physics: "Deceptive pseudorange injection pushes a single tractor off its guidance line (XTE divergence) while C/N0 signal health and RTK link stay completely normal.",
        why: "Differential XTE detection + backward ramp tracing catches gradual spoofing ramps and locks onto the true onset minute.",
        code: `# Channel 3: SPOOFING — single-tractor XTE divergence + backward onset trace
spf_mask = (
    (piv[("moving", tr)] > 1.0) & (piv[("link", tr)] > 0.8)
    & ((cn0_tr > 41.0) | ((cn0_tr - cn0_oth).abs() < 2.2))
    & (xte_tr > xte_thresh) & (xte_tr > xte_oth + diff_thresh)
)`
      }},
      {{
        cls: "EQUIPMENT_FAULT", title: "Channel 4: Equipment Fault",
        physics: "Hardware degradation isolates to ONE tractor (either RTK radio link dropout < 0.95 or antenna front-end C/N0 dropouts) while neighbors stay healthy.",
        why: "Dual sub-channels isolate single-machine RTK modem dropouts and LNA/antenna fades without false-triggering during fleet-wide interference.",
        code: `# Channel 4: EQUIPMENT_FAULT — single-machine isolation
link_mask = (piv[("moving", tr)] > 1.0) & (piv[("link", tr)] < 0.95) & (oth_link_min > 0.98)
ant_mask  = (~jam_active_mask) & (s4_vec < 0.15) & (piv[("min_cn0", tr)] < 24.0) & ((oth_cn0_mean - piv[("cn0", tr)]) > 0.6)`
      }},
      {{
        cls: "MULTIPATH", title: "Channel 5: Multipath Clustering",
        physics: "Brief (8-40s) single-satellite C/N0 notches and XTE spikes caused by stationary obstacles repeating at the exact same coordinates.",
        why: "Storm-aware 1-Hz notch extraction + latitude-adjusted 45m geodesic radius clustering discovers multipath with zero hardcoded GPS coords.",
        code: `# Channel 5: MULTIPATH — storm-aware 1-Hz notch + geodesic 45m spatial recurrence
m_per_deg_lon = 111320.0 * np.cos(np.radians(c["lat"]))
d_lat = (lats - c["lat"]) * 111320.0
d_lon = (lons - c["lon"]) * m_per_deg_lon
hits = int(np.sum((d_lat**2 + d_lon**2) <= 45.0**2))`
      }}
    ];

    const chListEl = document.getElementById("channelList");
    const chDetEl = document.getElementById("channelDetail");
    function renderChannel(ch, idx) {{
      document.querySelectorAll("#channelList .ep-item").forEach((el, i) => el.classList.toggle("active", i === idx));
      chDetEl.innerHTML = `
        <h2><span><span class="badge b-${{ch.cls}}">${{ch.cls}}</span> &nbsp;${{ch.title}}</span></h2>
        <div class="forensic-box" style="margin-bottom:14px;">
          <p style="margin:0 0 10px;"><strong>Physical Signature:</strong> ${{ch.physics}}</p>
          <p style="margin:0;"><strong>Architectural Advantage:</strong> ${{ch.why}}</p>
        </div>
        <h2 style="margin-top:14px;">Production Implementation (submission/solution.py)</h2>
        <pre class="code-block">${{ch.code}}</pre>
      `;
    }}
    CHANNELS.forEach((ch, idx) => {{
      const div = document.createElement("div");
      div.className = `ep-item c-${{ch.cls}}`;
      div.innerHTML = `<div class="ep-top"><span><span class="badge b-${{ch.cls}}">${{ch.cls}}</span></span></div><div class="ep-time">${{ch.title}}</div>`;
      div.onclick = () => renderChannel(ch, idx);
      chListEl.appendChild(div);
    }});
    renderChannel(CHANNELS[0], 0);
  </script>
</body>
</html>
"""
    out_html.write_text(html, encoding="utf-8")
    print(f"Generated 3-Tab Interactive Presentation App at {out_html}")


if __name__ == "__main__":
    build_dashboard(Path("data"), Path("submission.json"), Path("operator_dashboard.html"))