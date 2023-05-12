import pandas as pd
from pathlib import Path
from torch import nn
from fastai.data.core import DataLoaders
import torchapp as ta
from torchapp.examples.image_classifier import ImageClassifier
from typing import List
from pathlib import Path
from fastai.data.block import DataBlock, CategoryBlock
from fastai.data.transforms import ColReader, ColSplitter
from fastai.vision.data import ImageBlock
from fastai.vision.augment import Resize, ResizeMethod

from rich.console import Console
console = Console()

class CatenaMiner(ImageClassifier):
    """
    Identifies frame catena from manuscript images.
    """
    def dataloaders(
        self,
        csv: Path = ta.Param(default=None, help="A CSV with image paths and categories."),
        image_column: str = ta.Param(default="image", help="The name of the column with the image paths."),
        category_column: str = ta.Param(
            default="category", help="The name of the column with the category of the image."
        ),
        base_dir: Path = ta.Param(default="./", help="The base directory for images with relative paths."),
        validation_column: str = ta.Param(
            default="validation",
            help="The column in the dataset to use for validation. "
            "If the column is not in the dataset, then a validation set will be chosen randomly according to `validation_proportion`.",
        ),
        validation_value: str = ta.Param(
            default=None,
            help="If set, then the value in the `validation_column` must equal this string for the item to be in the validation set. "
        ),
        batch_size: int = ta.Param(default=16, help="The number of items to use in each batch."),
        width: int = ta.Param(default=224, help="The width to resize all the images to."),
        height: int = ta.Param(default=224, help="The height to resize all the images to."),
        resize_method: str = ta.Param(default="squish", help="The method to resize images."),
    ):
        df = pd.read_csv(csv)
        
        # Create splitter for training/validation images
        if validation_value is not None:
            validation_column_new = f"{validation_column} is {validation_value}"
            df[validation_column_new] = df[validation_column].astype(str) == validation_value
            validation_column = validation_column_new
            
        splitter = ColSplitter(validation_column)

        datablock = DataBlock(
            blocks=[ImageBlock, CategoryBlock],
            get_x=PathColReader(column_name=image_column, base_dir=base_dir),
            get_y=ColReader(category_column),
            splitter=splitter,
            item_tfms=Resize((height, width), method=resize_method),
        )

        return datablock.dataloaders(df, bs=batch_size)

