from pathlib import Path
from torch import nn
from fastai.data.core import DataLoaders
import torchapp as ta
from rich.console import Console
console = Console()

class CatenaMiner(ta.TorchApp):
    """
    Identifies frame catena from manuscript images.
    """

