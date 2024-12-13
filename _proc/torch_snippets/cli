from typer import Typer
from .loader import show as _show

cli = Typer()


@cli.command()
def time():
    """print the time now, in %Y-%m-%d %H:%M:%S format"""
    import time

    print(time.strftime("%Y-%m-%d %H:%M:%S"))


# Other cool utils


@cli.command()
def show(path):
    """print the object"""
    _show(path)
