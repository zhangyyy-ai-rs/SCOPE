"""CSV/row interface for the SCOPE support selector."""
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Dict, Iterable, Optional

from .metadata import DATASET_ORDER, DISPLAY_BLOCK, FORBIDDEN_DIAGNOSTIC_HINTS, ROUTE_INFO
from .rules import select_route


def load_diagnostics(path: Path) -> Dict[str, dict]:
    out: Dict[str, dict] = {}
    with Path(path).open(newline='') as f:
        reader = csv.DictReader(f)
        bad = [c for c in (reader.fieldnames or []) if any(h in c.lower() for h in FORBIDDEN_DIAGNOSTIC_HINTS)]
        if bad:
            raise ValueError(f'Forbidden result/oracle columns in diagnostic CSV: {bad}')
        for row in reader:
            out[row['dataset']] = row
    return out


def build_route_rows(diag: Dict[str, dict], datasets: Optional[Iterable[str]] = None) -> list[dict]:
    rows = []
    for ds in (list(datasets) if datasets else DATASET_ORDER):
        if ds not in diag:
            continue
        route, rule_id, reason = select_route(diag[ds])
        block, display = DISPLAY_BLOCK.get(ds, ('Other', ds))
        info = ROUTE_INFO[route]
        rows.append({
            'block': block,
            'dataset': ds,
            'display': display,
            'scope_route': route,
            'operator': info['operator'],
            'support_rule_id': rule_id,
            'support_reason': reason,
            'paper_role': info['role'],
        })
    return rows


def export_routes(rows: list[dict], json_out: Path | None = None, csv_out: Path | None = None) -> None:
    if json_out:
        Path(json_out).write_text(json.dumps({
            'version': 'support_only_v2_20260601',
            'protocol': 'support-only diagnostics; no validation/test/oracle route lookup',
            'routes': {r['dataset']: r['scope_route'] for r in rows},
            'details': rows,
        }, indent=2) + '\n')
    if csv_out and rows:
        with Path(csv_out).open('w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
