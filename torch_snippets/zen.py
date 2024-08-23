from .cli import cli


@cli.command()
def zen_of(language):
    if language == "python":
        import this
    else:
        raise NotImplementedError(f"zen of {language} is yet to be found")
