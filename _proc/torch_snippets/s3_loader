from .cli import cli
from .paths import P, parent, makedir, fname
from .load_defaults import exists
from .logger import Info, notify_waiting
from .misc import tryy


class S3Copier:
    def __init__(self):
        from s3fs import S3FileSystem

        self.s3 = S3FileSystem()

    @tryy
    def _download(self, s3_f, local_file):
        recursive = "." in s3_f
        with notify_waiting(f"Downloading\nSOURCE: {s3_f}\nDESTIN: {local_file}"):
            self.s3.download(
                str(s3_f), str(local_file), recursive=recursive
            )  # Download the object

    def download_s3_folder(
        self, s3_location, local_location, completeness_file, exclusion_list=None
    ):
        completeness_file = P(completeness_file)
        local_location = P(local_location)
        makedir(parent(local_location))
        if isinstance(exclusion_list, str):
            exclusion_list = [e.strip() for e in exclusion_list.split(",")]

        if not exists(completeness_file):
            objects = self.s3.listdir(s3_location)  # List objects in the S3 folder
            # objects = {_s: set(stems([k['Key'] for k in objects]))}
            objects = {fname(k["Key"]): k["Key"] for k in objects}
            for obj in objects:
                if exclusion_list and any(
                    excluded in obj for excluded in exclusion_list
                ):
                    continue  # Skip if the object matches any exclusion pattern
                s3_f = f"s3://{objects[obj]}"
                local_file = local_location / obj  # Define the local file path
                self._download(s3_f, local_file)
            completeness_file.touch()

    def download_s3_file(self, s3_location, local_location):
        makedir(parent(local_location))
        self.s3.download(s3_location, local_location)

    def upload_s3_folder(self, local_location, s3_location):
        makedir(parent(local_location))
        self.s3.upload(local_location, s3_location, recursive=True)

    def upload_s3_file(self, local_location, s3_location):
        makedir(parent(local_location))
        self.s3.upload(local_location, s3_location)


@cli.command()
def download_s3_folder(
    s3_location, local_location, completeness_file, exclusion_list=None
):
    S3Copier().download_s3_folder(
        s3_location, local_location, completeness_file, exclusion_list
    )


@cli.command()
def download_s3_file(s3_location, local_location):
    S3Copier().download_s3_file(s3_location, local_location)


@cli.command()
def upload_s3_folder(local_location, s3_location):
    S3Copier().upload_s3_folder(local_location, s3_location)


@cli.command()
def upload_s3_file(local_location, s3_location):
    S3Copier().upload_s3_file(local_location, s3_location)
