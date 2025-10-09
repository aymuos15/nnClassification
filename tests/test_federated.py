"""Tests for federated learning integration components."""

from __future__ import annotations

import builtins
import importlib
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.fixture
def flower_stub(monkeypatch):
    """Provide a lightweight Flower stub so modules can import successfully."""

    class DummyNumPyClient:  # pragma: no cover - simple container
        pass

    class DummyStrategyBase:  # pragma: no cover - simple container
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    class DummyFedAvg(DummyStrategyBase):
        pass

    class DummyFedProx(DummyStrategyBase):
        pass

    class DummyFedAdam(DummyStrategyBase):
        pass

    class DummyFedAdagrad(DummyStrategyBase):
        pass

    flwr_module = types.ModuleType("flwr")

    flwr_client = types.ModuleType("flwr.client")
    flwr_client.NumPyClient = DummyNumPyClient

    flwr_common = types.ModuleType("flwr.common")
    flwr_common.NDArrays = list
    flwr_common.Scalar = float
    flwr_common.Metrics = dict

    flwr_server_strategy = types.ModuleType("flwr.server.strategy")
    flwr_server_strategy.FedAvg = DummyFedAvg
    flwr_server_strategy.FedProx = DummyFedProx
    flwr_server_strategy.FedAdam = DummyFedAdam
    flwr_server_strategy.FedAdagrad = DummyFedAdagrad
    flwr_server_strategy.Strategy = DummyStrategyBase

    flwr_server = types.ModuleType("flwr.server")
    flwr_server.ServerConfig = lambda num_rounds: SimpleNamespace(num_rounds=num_rounds)
    flwr_server.start_server = lambda **kwargs: SimpleNamespace(**kwargs)
    flwr_server.strategy = flwr_server_strategy

    flwr_module.client = flwr_client
    flwr_module.common = flwr_common
    flwr_module.server = flwr_server

    monkeypatch.setitem(sys.modules, "flwr", flwr_module)
    monkeypatch.setitem(sys.modules, "flwr.client", flwr_client)
    monkeypatch.setitem(sys.modules, "flwr.common", flwr_common)
    monkeypatch.setitem(sys.modules, "flwr.server", flwr_server)
    monkeypatch.setitem(sys.modules, "flwr.server.strategy", flwr_server_strategy)

    for module_name in list(sys.modules):
        if module_name.startswith("ml_src.core.federated"):
            sys.modules.pop(module_name)

    core_module = importlib.import_module("ml_src.core.federated")
    client_module = importlib.import_module("ml_src.core.federated.client")
    server_module = importlib.import_module("ml_src.core.federated.server")
    partitioning_module = importlib.import_module("ml_src.core.federated.partitioning")

    return SimpleNamespace(
        core=core_module,
        client=client_module,
        server=server_module,
        partitioning=partitioning_module,
        FedAvg=DummyFedAvg,
        FedProx=DummyFedProx,
        FedAdam=DummyFedAdam,
        FedAdagrad=DummyFedAdagrad,
    )


def test_check_flower_available_missing(monkeypatch):
    """check_flower_available should raise if Flower cannot be imported."""

    original_import = builtins.__import__

    def fake_import(name, *args, **kwargs):  # pragma: no cover - simple guard
        if name.startswith("flwr"):
            raise ImportError("mock missing flwr")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    for module_name in list(sys.modules):
        if module_name.startswith("ml_src.core.federated"):
            sys.modules.pop(module_name)

    module = importlib.import_module("ml_src.core.federated")

    with pytest.raises(ImportError):
        module.check_flower_available()


def test_get_strategy_returns_expected_strategy(flower_stub):
    """get_strategy should instantiate the configured Flower strategy class."""

    config = {
        "federated": {
            "server": {
                "strategy": "FedAvg",
                "strategy_config": {
                    "fraction_fit": 0.5,
                    "fraction_evaluate": 0.5,
                    "min_fit_clients": 2,
                    "min_evaluate_clients": 2,
                    "min_available_clients": 2,
                },
            }
        }
    }

    strategy = flower_stub.server.get_strategy(config)

    assert isinstance(strategy, flower_stub.FedAvg)

    fit_metrics_fn = strategy.kwargs["fit_metrics_aggregation_fn"]
    aggregated = fit_metrics_fn(
        [
            (2, {"train_loss": 1.0, "train_acc": 0.5, "val_loss": 0.9, "val_acc": 0.6}),
            (6, {"train_loss": 0.5, "train_acc": 0.75, "val_loss": 0.4, "val_acc": 0.8}),
        ]
    )

    assert aggregated["train_loss"] == pytest.approx((2 * 1.0 + 6 * 0.5) / 8)
    assert aggregated["train_acc"] == pytest.approx((2 * 0.5 + 6 * 0.75) / 8)
    assert aggregated["val_loss"] == pytest.approx((2 * 0.9 + 6 * 0.4) / 8)
    assert aggregated["val_acc"] == pytest.approx((2 * 0.6 + 6 * 0.8) / 8)


def test_create_client_fn_applies_profile_overrides(flower_stub, monkeypatch, tmp_path):
    """Client profiles should override base training configuration."""

    created = {}

    class DummyClient:
        def __init__(self, config, client_id, run_dir):  # pragma: no cover - simple capture
            created["config"] = config
            created["client_id"] = client_id
            created["run_dir"] = run_dir

    monkeypatch.setattr(flower_stub.client, "FlowerClient", DummyClient)

    config = {
        "training": {
            "trainer_type": "standard",
            "batch_size": 16,
        },
        "federated": {
            "clients": {
                "profiles": [
                    {"id": [3], "trainer_type": "mixed_precision", "batch_size": 8},
                ]
            }
        },
    }

    client_fn = flower_stub.client.create_client_fn(config, run_base_dir=str(tmp_path))
    client_fn("3")

    assert created["client_id"] == 3
    assert created["run_dir"].endswith("fl_client_3")

    overrides = created["config"]["training"]
    assert overrides["trainer_type"] == "mixed_precision"
    assert overrides["batch_size"] == 8


def test_partition_data_label_skew_limits_classes(flower_stub):
    """Label-skew partitioning should limit the number of classes per client."""

    file_paths_by_class = {
        "cat": [f"raw/cat/cat_{idx}.jpg" for idx in range(4)],
        "dog": [f"raw/dog/dog_{idx}.jpg" for idx in range(4)],
        "bird": [f"raw/bird/bird_{idx}.jpg" for idx in range(4)],
    }

    partitions = flower_stub.partitioning.partition_data_label_skew(
        file_paths_by_class,
        num_clients=3,
        classes_per_client=2,
        seed=1,
    )

    for client_paths in partitions.values():
        class_names = {path.split("/")[1] for path in client_paths}
        assert len(class_names) <= 2


def test_create_federated_splits_creates_expected_files(tmp_path):
    """create_federated_splits should generate client train/val and shared test indices."""

    data_root = tmp_path / "dataset"
    raw_dir = data_root / "raw"
    for class_name in ("cat", "dog"):
        class_dir = raw_dir / class_name
        class_dir.mkdir(parents=True)
        for idx in range(5):
            (class_dir / f"{class_name}_{idx}.jpg").write_text("mock")

    splits_dir = data_root / "splits"

    from ml_src.core.federated.partitioning import create_federated_splits

    create_federated_splits(
        raw_data_dir=str(raw_dir),
        splits_dir=str(splits_dir),
        num_clients=2,
        partition_strategy="iid",
        train_val_split=0.75,
        test_split=0.2,
        seed=0,
    )

    client_train = splits_dir / "client_0_train.txt"
    client_val = splits_dir / "client_0_val.txt"
    client_train_1 = splits_dir / "client_1_train.txt"
    client_val_1 = splits_dir / "client_1_val.txt"
    test_file = splits_dir / "test.txt"

    for path in [client_train, client_val, client_train_1, client_val_1, test_file]:
        assert path.exists()

    def read_paths(path: Path) -> list[str]:
        return [line.strip() for line in path.read_text().splitlines() if line.strip()]

    all_paths = (
        read_paths(client_train)
        + read_paths(client_val)
        + read_paths(client_train_1)
        + read_paths(client_val_1)
        + read_paths(test_file)
    )

    assert len(all_paths) == 10
    assert len(set(all_paths)) == 10  # No duplicates across splits
