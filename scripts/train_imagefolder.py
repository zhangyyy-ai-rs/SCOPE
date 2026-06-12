#!/usr/bin/env python3
"""Train SCOPE on an ImageFolder dataset.

Expected layout:
  data_root/train/<class>/*.jpg
  data_root/val/<class>/*.jpg

The automatic route selector uses only frozen features from the training/support
split. The validation split is used only for reporting model accuracy after the
route has already been selected.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import timm
import torch

from scope.data import build_imagefolder_loaders
from scope.diagnostics import summarize_support
from scope.models.features import collect_frozen_features
from scope.models import InjectionConfig, inject_scope, trainable_parameter_count
from scope.routing.rules import select_route
from scope.training import evaluate, train_one_epoch


def write_diag(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        writer.writeheader()
        writer.writerow(row)


def main() -> None:
    parser = argparse.ArgumentParser(description='Train SCOPE with support-only route selection.')
    parser.add_argument('--data-root', required=True)
    parser.add_argument('--dataset-name', default='custom')
    parser.add_argument('--model', default='vit_base_patch16_224')
    parser.add_argument('--pretrained', action='store_true')
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight-decay', type=float, default=5e-4)
    parser.add_argument('--rank', type=int, default=8)
    parser.add_argument('--layers', default='all')
    parser.add_argument('--route', default='auto')
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--out-dir', default='runs/scope_imagefolder')
    parser.add_argument('--max-diag-batches', type=int, default=20)
    args = parser.parse_args()

    device = torch.device(args.device)
    out_dir = Path(args.out_dir)
    train_loader, val_loader, num_classes = build_imagefolder_loaders(
        args.data_root, batch_size=args.batch_size, workers=args.workers
    )

    frozen = timm.create_model(args.model, pretrained=args.pretrained, num_classes=num_classes).to(device)
    train_feat = collect_frozen_features(
        frozen, train_loader, device, max_batches=args.max_diag_batches, keep_images=True
    )
    diag = summarize_support(
        dataset=args.dataset_name,
        train_cls=train_feat.cls,
        train_y=train_feat.y,
        train_patch_mean=train_feat.patch_mean,
        train_patch_var=train_feat.patch_var,
        train_images_nchw=train_feat.images,
    )
    diag_row = diag.to_row()
    route, rule_id, reason = select_route(diag_row) if args.route == 'auto' else (args.route, 'manual', 'manual route override')
    write_diag(out_dir / 'support_diagnostics.csv', diag_row)

    model = timm.create_model(args.model, pretrained=args.pretrained, num_classes=num_classes).to(device)
    model = inject_scope(model, InjectionConfig(route=route, rank=args.rank, layers=args.layers))
    trainable = trainable_parameter_count(model)
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.amp.GradScaler('cuda', enabled=(device.type == 'cuda'))

    best_acc, best_epoch = 0.0, -1
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / 'log.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['epoch', 'loss', 'val_acc', 'best_acc', 'route', 'trainable_params'])
        writer.writeheader()
        for epoch in range(args.epochs):
            loss = train_one_epoch(model, train_loader, optimizer, device, scaler=scaler, amp=(device.type == 'cuda'))
            val_acc = evaluate(model, val_loader, device)
            if val_acc > best_acc:
                best_acc, best_epoch = val_acc, epoch
                torch.save({'model': model.state_dict(), 'route': route, 'diag': diag_row}, out_dir / 'best.pt')
            writer.writerow({'epoch': epoch, 'loss': loss, 'val_acc': val_acc, 'best_acc': best_acc, 'route': route, 'trainable_params': trainable})
            f.flush()
            print(f'epoch={epoch:03d} loss={loss:.4f} val_acc={val_acc:.3f} best={best_acc:.3f} route={route} params={trainable}')

    result = {
        'dataset': args.dataset_name,
        'route': route,
        'support_rule_id': rule_id,
        'support_reason': reason,
        'best_acc': best_acc,
        'best_epoch': best_epoch,
        'trainable_params': trainable,
        'epochs': args.epochs,
        'model': args.model,
        'rank': args.rank,
        'layers': args.layers,
        'protocol': 'route selected from train/support diagnostics only; validation used after selection for reporting',
    }
    (out_dir / 'result.json').write_text(json.dumps(result, indent=2) + "\n")
    print(f'SCOPE route: {route} ({rule_id}: {reason})')
    print(f'Best acc: {best_acc:.3f} at epoch {best_epoch}; trainable params: {trainable}')


if __name__ == '__main__':
    main()
