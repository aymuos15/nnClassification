import torch
import torch.nn as nn

from ml_src.core.network import load_model


def test_load_model_with_ema(tmp_path):
    """load_model should prioritize EMA weights when requested."""

    base_model = nn.Linear(4, 2)
    with torch.no_grad():
        for param in base_model.parameters():
            param.fill_(1.0)

    checkpoint_path = tmp_path / "checkpoint.pt"

    ema_state = {
        key: torch.full_like(value, 2.0)
        for key, value in base_model.state_dict().items()
    }

    torch.save(
        {
            "model_state_dict": base_model.state_dict(),
            "ema_state": {"ema_model_state": ema_state},
        },
        checkpoint_path,
    )

    target_model = nn.Linear(4, 2)
    loaded_model = load_model(target_model, str(checkpoint_path), torch.device("cpu"), use_ema=True)

    for param in loaded_model.state_dict().values():
        assert torch.allclose(param, torch.full_like(param, 2.0))


def test_load_model_without_ema(tmp_path):
    """Standard loading path remains unchanged."""

    base_model = nn.Linear(4, 2)
    with torch.no_grad():
        for param in base_model.parameters():
            param.fill_(3.0)

    checkpoint_path = tmp_path / "checkpoint.pt"

    torch.save({"model_state_dict": base_model.state_dict()}, checkpoint_path)

    target_model = nn.Linear(4, 2)
    loaded_model = load_model(target_model, str(checkpoint_path), torch.device("cpu"))

    for param in loaded_model.state_dict().values():
        assert torch.allclose(param, torch.full_like(param, 3.0))
