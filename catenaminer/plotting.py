from pathlib import Path
import pandas as pd
import typer

app = typer.Typer()


def plot_df(
    df,
    png:Path = None,
    html:Path = None,
    svg:Path = None,
    show:bool = False,
    thumbnails:bool = True,
    plot_width:int = 1000,
    plot_height:int = 600,
):
    if not html and not show:
        thumbnails = False

    from bokeh.plotting import figure, output_file

    if html:
        output_file(html)
    
    TOOLTIPS = """
    <div>
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
    plot = figure(width=plot_width, height=plot_height, tooltips=TOOLTIPS)
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
    png:Path = None,
    html:Path = None,
    svg:Path = None,
    show:bool = False,
    thumbnails:bool = True,
    plot_width:int = 1000,
    plot_height:int = 600,
):
    df = pd.read_csv(csv)
    return plot_df(
        df,
        png=png,
        html=html,
        svg=svg,
        show=show,
        thumbnails=thumbnails,
        plot_width=plot_width,
        plot_height=plot_height,
    )