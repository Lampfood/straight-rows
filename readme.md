# Straight Rows: Autonomous Fleet GNSS Diagnostics

## Overview
This repository contains a multi-channel GNSS anomaly detection and attribution engine developed for the Bluestaq hackathon. The system processes tractor sensor telemetry and Space Domain Awareness (SDA) environmental feeds to detect, classify, and attribute GNSS hazards and vehicle faults in real time. It outputs calibrated confidence scores and structured forensic evidence for agricultural operator decision support.

---

## Setup & Prerequisites

### Pinned Dependencies
The solution relies on standard data science libraries pinned to compatible versions:
- `python >= 3.10`
- `pandas == 2.2.0`
- `numpy == 1.26.3`

### Required Data Assets
Place the competition dataset inside a local `data/` directory at the root of the repository. The required folder structure is:
```text
data/
├── farm/
│   └── tractor_telemetry.csv.gz
├── sda/
│   ├── sgi_indices.json
│   ├── scintillation_series.json
│   ├── constellation_advisories.json
│   └── rfi_reports.json
└── labels/
    └── train_episodes.json
```

---

## Execution Commands

### 1. Regenerate and Validate the Submission File
To execute the core detection engine, evaluate performance against the development set, generate the blind evaluation predictions, and validate the output schema, run:
```bash
python submission/solution.py --data data --output submission.json --github-handle Lampfood --repo-url https://github.com/Lampfood/straight-rows
python submission/validate_submission.py submission.json
```

### 2. Generate the Interactive Operator Dashboard
To build and launch the standalone operator presentation app:
```bash
python build_dashboard.py
```
Open `operator_dashboard.html` in any modern web browser to inspect live mission control, telemetry timelines, field tracks, and code architecture.

---

## Approach & Utilization of Labeled Data

### Multi-Channel Detection Architecture
Rather than utilizing a mutually exclusive decision tree, our engine runs five independent parallel detection channels. This guarantees that overlapping events—such as terrestrial radio jamming occurring during an active solar storm—are successfully isolated:
1. **SPACE_WEATHER:** Detects regional ionospheric scintillation ($S_4 \ge 0.15$) paired with fleet carrier-to-noise ($C/N_0$) depression while RTK base links remain intact.
2. **JAMMING:** Identifies localized RF interference using a dual-regime gate (clear-sky relative depression vs. storm-concurrent carrier and satellite collapse) affecting two or more tractors simultaneously.
3. **SPOOFING:** Detects single-tractor guidance divergence (cross-track error) while signal strength and RTK link health remain nominal, utilizing backward onset backtracking to pinpoint attack start times.
4. **EQUIPMENT_FAULT:** Isolates single-machine hardware degradation, including RTK radio correction link dropouts or onboard antenna front-end fading, while neighboring tractors remain healthy.
5. **MULTIPATH:** Operates on raw 1-Hz telemetry to extract brief signal notches ($8\text{s}$ to $40\text{s}$) and applies latitude-adjusted 45-meter spatial recurrence clustering to identify stationary physical reflectors (grain bins, tree lines, headland structures) with zero hardcoded GPS coordinates.

### How Labeled Data Was Used
The development set (`train_episodes.json`, covering May 4 through May 5) was used to benchmark detection thresholds and tune interval merging parameters. By comparing our model predictions against ground-truth labels using a local diagnostic scorer, we optimized temporal intersection-over-union (IoU) and onset timing to achieve a Dev score of 39.00 / 40.00 (0.96 Macro-F1 with 100% mean onset credit).

---

## Failure Modes & Edge Cases
- **Low-Power Single Jammer Ambiguity:** A localized jammer affecting only one tractor during completely quiet space weather can occasionally resemble an internal antenna hardware failure because both present as single-machine signal loss while peers remain healthy.
- **Gradual Spoofing Onset Latency:** Extremely slow, stealthy spoofing attacks that ramp position error up at a microscopic rate (<0.12 m/min) require a brief period of spatial divergence before crossing the cross-track error detection threshold.

---

## Additional Data & Future Sensor Roadmap
To push attribution accuracy even further in future iterations, the following sensors and data feeds would provide immediate value:
1. **Receiver AGC (Automatic Gain Control) Telemetry:** Tracks how hard the receiver hardware is working to amplify incoming signals. Natural space weather fading reduces signal power without driving up receiver noise power, whereas man-made RF jamming floods the receiver with high-power noise, instantly separating natural ionospheric fading from terrestrial jamming.
2. **Dual-Antenna GNSS Heading:** Measuring physical heading vectors directly on the tractor roof would provide instant proof of spoofing the moment satellite-derived heading diverges from wheel or IMU steering.