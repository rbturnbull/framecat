from torchapp.testing import TorchAppTestCase
from catenaminer.apps import CatenaMiner


class TestCatenaMiner(TorchAppTestCase):
    app_class = CatenaMiner
