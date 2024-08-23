from .cli import cli
from .paths import P, parent, makedir
from .load_defaults import exists


@cli.command()
def download_s3_folder(s3_location, local_location, completeness_file):
    from s3fs import S3FileSystem

    s3 = S3FileSystem()
    completeness_file = P(completeness_file)
    makedir(parent(local_location))
    if not exists(completeness_file):
        s3.download(s3_location, local_location, recursive=True)
        completeness_file.touch()
