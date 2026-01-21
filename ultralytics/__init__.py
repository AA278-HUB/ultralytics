# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

__version__ = "8.3.218"

import importlib
import os
from typing import TYPE_CHECKING

# Set ENV variables (place before imports)
if not os.environ.get("OMP_NUM_THREADS"):
    os.environ["OMP_NUM_THREADS"] = "1"  # default for reduced CPU utilization during training

from ultralytics.utils import ASSETS, SETTINGS
from ultralytics.utils.checks import check_yolo as checks
from ultralytics.utils.downloads import download

settings = SETTINGS

MODELS = ("YOLO", "YOLOWorld", "YOLOE", "NAS", "SAM", "FastSAM", "RTDETR")

__all__ = (
    "__version__",
    "ASSETS",
    *MODELS,
    "checks",
    "download",
    "settings",
)

if TYPE_CHECKING:
    # Enable hints for type checkers
    from ultralytics.models import YOLO, YOLOWorld, YOLOE, NAS, SAM, FastSAM, RTDETR  # noqa


def __getattr__(name: str):
    """Lazy-import model classes on first access."""
    if name in MODELS:
        return getattr(importlib.import_module("ultralytics.models"), name)
    raise AttributeError(f"module {__name__} has no attribute {name}")


def __dir__():
    """Extend dir() to include lazily available model names for IDE autocompletion."""
    return sorted(set(globals()) | set(MODELS))


if __name__ == "__main__":
    print(__version__)
import torch.nn as nn

if self.Distillation is not None:

    distill_weight = ((1 - math.cos(i * math.pi / len(self.train_loader))) / 2) * (0.1 - 1) + 1

    with torch.no_grad():

        pred = self.Distillation(batch['img'])

    self.d_loss = distillation_loss.get_loss()

    self.d_loss *= distill_weight

    if i == 0:
        print(self.d_loss, '-----------------')

        print(self.loss, '-----------------')

self.loss += self.d_loss