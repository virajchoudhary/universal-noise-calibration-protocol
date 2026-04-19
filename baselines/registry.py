"""
Baseline method registry for UNCP comparison experiments.

Provides a unified factory for creating baseline trainers
and a convenience function for running all baselines.
"""

from typing import Dict, Any, List, Optional

from baselines.erm import ERMTrainer
from baselines.group_dro import GroupDROTrainer
from baselines.jtt import JTTTrainer
from baselines.mixup import MixupTrainer
from baselines.cutmix import CutMixTrainer
from baselines.adversarial_training import AdversarialTrainer
from baselines.dropout_baseline import DropoutBaselineTrainer


# Registry mapping method names to their trainer classes
BASELINE_REGISTRY: Dict[str, type] = {
    "erm": ERMTrainer,
    "group_dro": GroupDROTrainer,
    "jtt": JTTTrainer,
    "mixup": MixupTrainer,
    "cutmix": CutMixTrainer,
    "adversarial": AdversarialTrainer,
    "dropout_0.1": None,  # Special-cased: same class, different dropout_rate
    "dropout_0.3": None,
    "dropout_0.5": None,
}

# Methods that do NOT require group labels during training
NO_GROUP_LABEL_METHODS = {"erm", "jtt", "mixup", "cutmix", "adversarial",
                           "dropout_0.1", "dropout_0.3", "dropout_0.5", "uncp"}

# Methods that DO require group labels during training
GROUP_LABEL_METHODS = {"group_dro"}


def get_baseline_names() -> List[str]:
    """Return list of all registered baseline method names."""
    return list(BASELINE_REGISTRY.keys())


def get_baseline(
    name: str,
    model,
    train_loader,
    val_loader,
    test_loader,
    config,
    device: str = "cpu",
):
    """Create a baseline trainer by name.

    Parameters
    ----------
    name : str
        Method name (e.g., 'erm', 'group_dro', 'dropout_0.3').
    model : nn.Module
        The neural network model.
    train_loader, val_loader, test_loader : DataLoader
        Data loaders.
    config : DictConfig
        Configuration.
    device : str
        Device string.

    Returns
    -------
    Trainer instance with a .run() method.
    """
    if name.startswith("dropout_"):
        rate = float(name.split("_")[1])
        return DropoutBaselineTrainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            config=config,
            dropout_rate=rate,
            device=device,
        )

    if name not in BASELINE_REGISTRY or BASELINE_REGISTRY[name] is None:
        raise ValueError(
            f"Unknown baseline: {name}. "
            f"Available: {get_baseline_names()}"
        )

    cls = BASELINE_REGISTRY[name]
    return cls(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        config=config,
        device=device,
    )