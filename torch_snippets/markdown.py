try:
    from markitdown import MarkItDown
except ImportError:
    MarkItDown = None
import os
from typer import Option
from .logger import Info
from .paths import P
from .cli import cli
from .misc import track2


def convert_to_markdown(input_file, output_dir, md_parser, root_path=None):
    """Converts a single file to markdown"""
    try:
        if root_path is None:
            output_file_path = output_dir / f"{input_file.stem}.md"
        else:
            relative_path = input_file.relative_to(root_path)
            output_file_path = output_dir / relative_path.with_suffix(f"{input_file.suffix}.md")

        if os.path.exists(output_file_path):
            Info(f"File {output_file_path} already exists, skipping.")
            return None

        result = md_parser.convert(str(input_file))
        output_file_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file_path, "w") as f:
            f.write(result.text_content)
        return f"Converted {input_file} to {output_file_path}"
    except Exception as e:
        return f"Could not convert {input_file}. Reason: {e}"


@cli.command('mkd')
def create_markdown(input_path: str, output_path: str = Option(None, "-o", "--output")):
    """
    Recursively creates markdown files from a source folder or a single file to a destination folder.
    """
    input_path = P(input_path).resolve()

    if output_path is None:
        output_path = input_path

    output_path = P(output_path)

    if not output_path.exists():
        output_path.mkdir(parents=True)

    try:
        md = MarkItDown()
    except:
        raise ImportError("Please install the 'markitdown' package to use this command. You can install it with 'pip install markitdown[all]'.")

    if input_path.is_file():
        if not output_path.is_dir():
            output_path = output_path.parent
        result = convert_to_markdown(input_path, output_path, md)
        print(result)
        return

    for root, _, files in os.walk(input_path):
        tracker = track2(files)
        for file in tracker:
            file_path = P(root) / file
            result = convert_to_markdown(file_path, output_path, md, root_path=input_path)
            if result is None:
                continue
            if "Could not convert" in result:
                print(result)
            else:
                tracker.send(result)

