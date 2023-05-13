from pathlib import Path
import pandas as pd
import typer
import os
import base64
from io import BytesIO
from PIL import Image

app = typer.Typer()


def generate_thumbnail(path, width, height):
    im = Image.open(path)
    size = width, height
    im.thumbnail(size, Image.ANTIALIAS)
    buffered = BytesIO()
    im.save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode("ascii")


def plot_df(
    df,
    title:str="",
    png:Path = None,
    html:Path = None,
    svg:Path = None,
    show:bool = False,
    thumbnails:bool = True,
    plot_width:int = 1000,
    plot_height:int = 600,
    thumbnail_size:int = 100,
):
    from bokeh.plotting import figure, output_file

    if html:
        output_file(html)
    

    if not html and not show:
        thumbnails = False

    thumbnail_tooltip = ""
    if thumbnails:
        df["thumbnail"] = df.path.apply(lambda path: generate_thumbnail(path, thumbnail_size, thumbnail_size))
        thumbnail_tooltip = '<img src="data:image/png;base64, @thumbnail{safe}" alt="Thumbnail" />'

    TOOLTIPS = f"""
    <div>
        {thumbnail_tooltip}
        <p>@path</p>
        <p>@prediction</p>
        <p>@frame</p>
    </div>
    """

    # imcluster_io.df["path"] = [str(x) for x in imcluster_io.images]
    # if not imcluster_io.has_column("thumbnail") or force or force_thumbnails:
    #     print(f"Generating thumbnails within box ({thumbnail_width}x{thumbnail_height})")
    #     imcluster_io.save_column(
    #         "thumbnail",
    #         imcluster_io.df.apply(lambda row: generate_thumbnail(row["path"], thumbnail_width, thumbnail_height), axis=1),
    #     )

    df.path = df['path'].apply(lambda x:str(x))

    if not title:
        paths = df.path.tolist()
        title = os.path.commonprefix(paths)

    plot = figure(title=title, width=plot_width, height=plot_height, tooltips=TOOLTIPS)
    plot.line("index", "frame", source=df)
    plot.circle("index", "frame", source=df, size=5)
    
    if png:
        from bokeh.io import export_png
        export_png(plot, filename=str(png))

    if show:
        from bokeh.plotting import show as show_plot
        show_plot(plot)

    if svg:
        from bokeh.io import export_svg
        plot.output_backend = "svg"
        export_svg(plot, filename=svg)
    
    return plot


@app.command()
def plot_csv(
    csv:Path,
    png:Path = typer.Option(None, help="Path to save output plot as PNG."), 
    html:Path = typer.Option(None, help="Path to save output plot as HTML."), 
    svg:Path = typer.Option(None, help="Path to save output plot as SVG."), 
    show:bool = typer.Option(False, help="Whether or not to show the plot."),
    thumbnails:bool = typer.Option(True, help="Whether or not to embed images of the thumbnails into the output."),
    plot_width:int = typer.Option(1000, help="The width of the output plot."),
    plot_height:int = typer.Option(600, help="The height of the output plot."),
    title:str = typer.Option("", help="The title of the plot. By default it takes the longest common prefix of the images."),
    thumbnail_size:int = typer.Option(200, help="The max width and height of the embedded thumbnails."),
):
    df = pd.read_csv(csv)
    return plot_df(
        df,
        title=title,
        png=png,
        html=html,
        svg=svg,
        show=show,
        thumbnails=thumbnails,
        plot_width=plot_width,
        plot_height=plot_height,
        thumbnail_size=thumbnail_size,
    )