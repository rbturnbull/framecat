import pandas as pd
from pathlib import Path
import torch
from torch import nn
import torchapp as ta
from torchapp.examples.image_classifier import ImageClassifier, PathColReader
from typing import List
from pathlib import Path
from fastai.data.block import DataBlock, CategoryBlock, TransformBlock
from fastai.data.transforms import ColReader, ColSplitter, DisplayedTransform
from fastai.vision.data import ImageBlock
from fastai.vision.augment import Resize
from fastai.vision.core import PILImage
# from fastai.metrics import accuracy, Precision, Recall, F1Score
from fastai.vision.augment import aug_transforms

from .plotting import plot_df
from .loss import FocalLoss, CatenaCombinedLoss
from .metrics import accuracy, PrecisionScore, RecallScore, F1Score, date_accuracy, DATE_MEAN, DATE_STD

from rich.console import Console
console = Console()




def normalise_date(date):
    return (float(date) - DATE_MEAN)/DATE_STD


class GetDateRange():    
    def __call__(self, item:Path):
        min_date = normalise_date(item["min_date"])
        max_date = normalise_date(item["max_date"])

        return torch.as_tensor([min_date,max_date])


class GrayscaleTransform(DisplayedTransform):
    def encodes(self,im:PILImage):
        return im.convert('L')            



class FrameCat(ImageClassifier):
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
        grayscale: bool = ta.Param(default=False, help="Whether to convert the images to grayscale."),
        date_weight:float = ta.Param(default=0.0, help="How much to use the date in the loss."),
        max_lighting:float=0.0,
        max_rotate:float=0.0,
        max_warp:float=0.2,
        max_zoom:float=1.0,
        do_flip:bool=False,
        p_affine:float=0.75,
        p_lighting:float=0.75,
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

        batch_transforms = aug_transforms(
            p_lighting=p_lighting,
            p_affine=p_affine,
            max_rotate=max_rotate, 
            do_flip=do_flip, 
            max_lighting=max_lighting, 
            max_zoom=max_zoom,
            max_warp=max_warp,
            pad_mode='zeros',
        )

        blocks = [ImageBlock, CategoryBlock]
        getters = [
            PathColReader(column_name=image_column, base_dir=base_dir),
            ColReader(category_column),
        ]

        self.date_weight = date_weight
        if date_weight > 0.0:
            blocks.append(TransformBlock)
            getters.append(GetDateRange())

        datablock = DataBlock(
            blocks=blocks,
            getters=getters,
            splitter=splitter,
            item_tfms=item_transforms,
            batch_tfms=batch_transforms,
            n_inp=1,
        )

        dls = datablock.dataloaders(df, bs=batch_size)

        if date_weight > 0.0:
            dls.c += 1

        return dls

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
        metrics = [accuracy, PrecisionScore(), RecallScore(), F1Score()]
        if self.date_weight > 0.0:
            metrics.append(date_accuracy)
        return metrics

    def loss_func(self, gamma:float=0.0):
        if self.date_weight > 0.0:
            return CatenaCombinedLoss(date_weight=self.date_weight, gamma=gamma)

        if gamma == 0.0:
            return nn.CrossEntropyLoss()
        else:
            return FocalLoss(gamma=gamma)
