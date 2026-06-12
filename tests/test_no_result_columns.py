import csv
from pathlib import Path

import pytest

from scope.routing.selector import load_diagnostics


def test_oracle_columns_rejected(tmp_path: Path):
    path = tmp_path / 'bad.csv'
    with path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['dataset', 'oracle_route', 'cls_effective_rank'])
        writer.writeheader()
        writer.writerow({'dataset': 'x', 'oracle_route': 'bad', 'cls_effective_rank': 1})
    with pytest.raises(ValueError):
        load_diagnostics(path)


def test_support_diagnostic_column_is_allowed(tmp_path: Path):
    path = tmp_path / 'ok.csv'
    with path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['dataset', 'nearest_centroid_support_acc', 'cls_effective_rank'])
        writer.writeheader()
        writer.writerow({'dataset': 'x', 'nearest_centroid_support_acc': 88, 'cls_effective_rank': 10})
    assert 'x' in load_diagnostics(path)
