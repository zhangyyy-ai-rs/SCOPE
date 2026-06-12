"""Command-line interface for SCOPE route selection."""
from __future__ import annotations

import argparse
from pathlib import Path

from .selector import build_route_rows, export_routes, load_diagnostics


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description='Select SCOPE route from support-only diagnostics.')
    parser.add_argument('--diag', type=Path, default=Path('configs/support_diagnostics_vtab19.csv'))
    parser.add_argument('--dataset')
    parser.add_argument('--json-out', type=Path)
    parser.add_argument('--csv-out', type=Path)
    args = parser.parse_args(argv)

    diag = load_diagnostics(args.diag)
    rows = build_route_rows(diag, [args.dataset] if args.dataset else None)
    if args.dataset:
        print(rows[0]['scope_route'])
    export_routes(rows, args.json_out, args.csv_out)


if __name__ == '__main__':
    main()
