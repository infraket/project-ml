from pathlib import Path
import click

from .data import get_dataset


@click.command()
@click.option(

    "-d",
    "--dataset-path",
    default="./data/train.csv",

    type=click.Path(path_type=Path),
    # type=click.Path(exists=True, dir_okay=False, path_type=Path),
    show_default=True,
)
@click.option(
    "-eda",
    "--eda-profiling",
    type=bool,
    default=False, )
def train(dataset_path: Path, eda_profiling):
    get_dataset(dataset_path, eda_profiling)
