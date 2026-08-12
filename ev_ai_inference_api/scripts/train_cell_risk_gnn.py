from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import random
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.schemas.twins import TwinFrame
from app.services.cell_risk_gnn import (
    FEATURE_NAMES,
    MODEL_ID,
    frame_features,
    graph_adjacency,
)


class CellRiskGraphSAGE(nn.Module):
    """Training form of the NumPy GraphSAGE production runtime."""

    def __init__(self, feature_count: int, hidden_dim: int = 24) -> None:
        super().__init__()
        self.self_layer_1 = nn.Linear(feature_count, hidden_dim)
        self.neighbor_layer_1 = nn.Linear(feature_count, hidden_dim, bias=False)
        self.self_layer_2 = nn.Linear(hidden_dim, hidden_dim)
        self.neighbor_layer_2 = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.output = nn.Linear(hidden_dim, 4)
        self.register_buffer(
            "adjacency",
            torch.from_numpy(graph_adjacency()),
        )

    def _neighbor_mean(self, values: torch.Tensor) -> torch.Tensor:
        return torch.einsum("ij,bjf->bif", self.adjacency, values)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        neighbors = self._neighbor_mean(features)
        hidden = torch.relu(
            self.self_layer_1(features) + self.neighbor_layer_1(neighbors)
        )
        hidden_neighbors = self._neighbor_mean(hidden)
        hidden = torch.relu(
            self.self_layer_2(hidden)
            + self.neighbor_layer_2(hidden_neighbors)
        )
        return self.output(hidden)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the 96-cell GraphSAGE current-risk model"
    )
    parser.add_argument("--scenario-dir", type=Path, default=Path("runtime/scenarios"))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("model_bundles/cell_risk_gnn_v1"),
    )
    parser.add_argument("--stride", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=18)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--hidden-dim", type=int, default=24)
    parser.add_argument("--seed", type=int, default=20260812)
    return parser.parse_args()


def _targets(frame: TwinFrame) -> np.ndarray:
    # Preserve the validated sensor/image cell labels used by the current twin.
    # The graph model learns their spatial context; no future frame is a target.
    return np.asarray(frame.state_level, dtype=np.int64)


def _thermal_augmented(frame: TwinFrame) -> TwinFrame:
    """Add a scenario-aligned thermal AI result for fusion-path training."""

    targets = _targets(frame)
    temperatures = np.asarray(frame.temperature_decic, dtype=np.float32)
    voltages = np.asarray(frame.voltage_mv, dtype=np.float32)
    voltage_median = float(np.median(voltages))
    hotspot = max(
        range(len(targets)),
        key=lambda index: (
            int(targets[index]),
            float(temperatures[index]),
            abs(float(voltages[index]) - voltage_median),
        ),
    )
    return frame.model_copy(
        update={
            "image_risk_level": int(targets.max()),
            "image_confidence": 0.92,
            "image_model_status": "ready",
            "hotspot_cell_index": hotspot,
        }
    )


def load_dataset(
    scenario_dir: Path,
    stride: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str]]:
    if stride <= 0:
        raise ValueError("stride must be positive")
    train_features: list[np.ndarray] = []
    train_targets: list[np.ndarray] = []
    validation_features: list[np.ndarray] = []
    validation_targets: list[np.ndarray] = []
    sources: list[str] = []
    for frames_path in sorted(scenario_dir.glob("*/frames.jsonl.gz")):
        sampled: list[TwinFrame] = []
        with gzip.open(frames_path, "rt", encoding="utf-8") as handle:
            for index, line in enumerate(handle):
                if index % stride == 0 and line.strip():
                    sampled.append(TwinFrame.model_validate_json(line))
        if len(sampled) < 5:
            continue
        split = max(1, int(len(sampled) * 0.8))
        for destination_x, destination_y, frames in (
            (train_features, train_targets, sampled[:split]),
            (validation_features, validation_targets, sampled[split:]),
        ):
            for frame in frames:
                target = _targets(frame)
                destination_x.append(frame_features(frame))
                destination_y.append(target)
                destination_x.append(frame_features(_thermal_augmented(frame)))
                destination_y.append(target)
        sources.append(frames_path.parent.name)
    if not train_features or not validation_features:
        raise ValueError(f"no scenario frames found under {scenario_dir}")
    return (
        np.stack(train_features),
        np.stack(train_targets),
        np.stack(validation_features),
        np.stack(validation_targets),
        sources,
    )


def evaluate(
    model: CellRiskGraphSAGE,
    features: torch.Tensor,
    targets: torch.Tensor,
) -> tuple[float, np.ndarray, np.ndarray]:
    predictions: list[np.ndarray] = []
    with torch.inference_mode():
        for offset in range(0, len(features), 256):
            predictions.append(
                model(features[offset : offset + 256]).argmax(dim=-1).numpy()
            )
    predicted = np.concatenate(predictions).reshape(-1)
    truth = targets.numpy().reshape(-1)
    macro_f1 = float(f1_score(truth, predicted, labels=[0, 1, 2, 3], average="macro"))
    return macro_f1, truth, predicted


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    train_x, train_y, validation_x, validation_y, sources = load_dataset(
        args.scenario_dir,
        args.stride,
    )
    train_features = torch.from_numpy(train_x)
    train_targets = torch.from_numpy(train_y)
    validation_features = torch.from_numpy(validation_x)
    validation_targets = torch.from_numpy(validation_y)
    counts = torch.bincount(train_targets.reshape(-1), minlength=4).float()
    class_weights = torch.sqrt(counts.sum() / counts.clamp_min(1.0))
    class_weights = class_weights / class_weights.mean()
    dataset = TensorDataset(train_features, train_targets)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(args.seed),
    )
    model = CellRiskGraphSAGE(len(FEATURE_NAMES), args.hidden_dim)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    best_f1 = -1.0
    best_state: dict[str, torch.Tensor] | None = None
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        for features, targets in loader:
            optimizer.zero_grad(set_to_none=True)
            logits = model(features)
            loss = criterion(logits.reshape(-1, 4), targets.reshape(-1))
            loss.backward()
            optimizer.step()
            total_loss += float(loss.detach()) * len(features)
        model.eval()
        macro_f1, _, _ = evaluate(model, validation_features, validation_targets)
        print(
            f"epoch={epoch + 1:02d} loss={total_loss / len(dataset):.6f} "
            f"validation_macro_f1={macro_f1:.6f}"
        )
        if macro_f1 > best_f1:
            best_f1 = macro_f1
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
    if best_state is None:
        raise RuntimeError("training did not produce a model")
    model.load_state_dict(best_state)
    model.eval()
    macro_f1, truth, predicted = evaluate(
        model,
        validation_features,
        validation_targets,
    )
    print(classification_report(truth, predicted, labels=[0, 1, 2, 3], digits=4))
    print(confusion_matrix(truth, predicted, labels=[0, 1, 2, 3]))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    model_path = args.output_dir / "model.npz"
    np.savez_compressed(
        model_path,
        feature_names=np.asarray(FEATURE_NAMES),
        self_1_weight=best_state["self_layer_1.weight"].numpy(),
        self_1_bias=best_state["self_layer_1.bias"].numpy(),
        neighbor_1_weight=best_state["neighbor_layer_1.weight"].numpy(),
        self_2_weight=best_state["self_layer_2.weight"].numpy(),
        self_2_bias=best_state["self_layer_2.bias"].numpy(),
        neighbor_2_weight=best_state["neighbor_layer_2.weight"].numpy(),
        output_weight=best_state["output.weight"].numpy(),
        output_bias=best_state["output.bias"].numpy(),
    )
    digest = hashlib.sha256(model_path.read_bytes()).hexdigest()
    label_counts = Counter(int(value) for value in train_y.reshape(-1))
    manifest = {
        "model_id": MODEL_ID,
        "architecture": "two-layer GraphSAGE node classifier",
        "task": "current 96-cell risk classification",
        "future_prediction": False,
        "feature_names": list(FEATURE_NAMES),
        "classes": ["normal", "caution", "warning", "danger"],
        "layout_id": "generic_ev_concept_96_v1",
        "training_source": "generated digital-twin scenario frames",
        "thermal_signal_augmentation": (
            "scenario-label-aligned image risk, confidence, and hotspot"
        ),
        "training_scenarios": sources,
        "training_label_counts": {str(key): value for key, value in sorted(label_counts.items())},
        "validation_macro_f1": macro_f1,
        "minimum_risk_probability": 0.50,
        "prototype_only": True,
        "safety_authority": False,
        "sha256": digest,
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"saved {model_path} sha256={digest} validation_macro_f1={macro_f1:.6f}")


if __name__ == "__main__":
    main()
