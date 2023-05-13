import pandas as pd
from pathlib import Path
from torch import nn
from fastai.data.core import DataLoaders
import torchapp as ta
from torchapp.examples.image_classifier import ImageClassifier, PathColReader
from typing import List
from pathlib import Path
from fastai.data.block import DataBlock, CategoryBlock
from fastai.data.transforms import ColReader, ColSplitter, DisplayedTransform
from fastai.vision.data import ImageBlock
from fastai.vision.augment import Resize, ResizeMethod
from fastai.vision.core import PILImage
from fastai.metrics import accuracy, Precision, Recall, F1Score

from .plotting import plot_df
from .loss import FocalLoss

from rich.console import Console
console = Console()


class GrayscaleTransform(DisplayedTransform):
    def encodes(self,im:PILImage):
        return im.convert('L')            



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
        grayscale: bool = ta.Param(default=True, help="Whether to convert the images to grayscale."),
    ):
        df = pd.read_csv(csv)
        
        # Create splitter for training/validation images
        if validation_value is not None:
            validation_column_new = f"{validation_column} is {validation_value}"
            df[validation_column_new] = df[validation_column].astype(str) == validation_value
            validation_column = validation_column_new
            
        splitter = ColSplitter(validation_column)
        item_transforms = [Resize((height, width), method=resize_method)]
        
        if grayscale:
            item_transforms.append(GrayscaleTransform())

        datablock = DataBlock(
            blocks=[ImageBlock, CategoryBlock],
            get_x=PathColReader(column_name=image_column, base_dir=base_dir),
            get_y=ColReader(category_column),
            splitter=splitter,
            item_tfms=item_transforms,
        )

        return datablock.dataloaders(df, bs=batch_size)

    def output_results(
        self, 
        results, 
        output_csv:Path = ta.Param(None, help="Path to write predictions in CSV format"), 
        verbose:bool = True, 
        png:Path = ta.Param(None, help="Path to save output plot as PNG."), 
        html:Path = ta.Param(None, help="Path to save output plot as HTML."), 
        svg:Path = ta.Param(None, help="Path to save output plot as SVG."), 
        show:bool = ta.Param(False, help="Whether or not to show the plot."),
        thumbnails:bool = ta.Param(True, help="Whether or not to embed images of the thumbnails into the output."),
        plot_width:int = ta.Param(1000, help="The width of the output plot."),
        plot_height:int = ta.Param(600, help="The height of the output plot."),
        title:str = ta.Param("", help="The title of the plot. By default it takes the longest common prefix of the images."),
        thumbnail_size:int = ta.Param(200, help="The max width and height of the embedded thumbnails."),
        **kwargs
    ):
        df = super().output_results(results, output_csv, verbose, **kwargs)

        if (png or html or svg or show):
            return plot_df(
                df,
                title=title,
                thumbnail_size=thumbnail_size,
                png=png,
                html=html,
                svg=svg,
                show=show,
                thumbnails=thumbnails,
                plot_width=plot_width,
                plot_height=plot_height,
            )

        return df

    def metrics(self):
        return [accuracy, Precision(average="macro"), Recall(average="macro"), F1Score(average="macro")]

    def loss_func(self,gamma:float=0.0):
        if gamma == 0.0:
            return nn.CrossEntropyLoss()
        else:
            return FocalLoss(gamma=gamma)
