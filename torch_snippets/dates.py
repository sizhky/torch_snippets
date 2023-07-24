import datetime

ALL_DATE_FORMATS = [
    "%d.%m.%Y",
    "%Y-%m-%d %H:%M:%S",
    "%Y%m%d",
    "%m-%d-%Y",
    "%d-%m-%Y",
    "%m/%d/%Y",
    "%d/%m/%Y",
    "%Y-%m-%d",
    "%b %d %Y",
    "%d %b %Y",
    "%d-%b-%Y",
    "%d/%b/%Y",
    "%b-%d-%Y",
    "%b/%d/%Y",
    "%m-%d-%y",
    "%d-%m-%y",
    "%m/%d/%y",
    "%d/%m/%y",
    "%y-%m-%d",
    "%b %d %y",
    "%d %b %y",
    "%d-%b-%y",
    "%d/%b/%y",
    "%b-%d-%y",
    "%b/%d/%y",
    "%b %d %y",
    "%d %b %y",
    "%d-%b-%y",
    "%d/%b/%y",
    "%b-%d-%y",
    "%b/%d/%y",
]


def make_uniform_date_format(value, target_fmt="%d.%m.%Y", mode="raise"):
    available_modes = ["raise", "debug", "return", "default"]
    if isinstance(value, datetime.datetime):
        return value.strftime(target_fmt)
    for fmt in ALL_DATE_FORMATS:
        try:
            if mode == "debug":
                return (
                    value,
                    datetime.datetime.strptime(value, fmt).strftime(target_fmt),
                    fmt,
                )
            return datetime.datetime.strptime(value, fmt).strftime(target_fmt)
        except:
            ...
    if mode == "raise":
        raise NotImplementedError(f"Unable to give a proper date for `{value}`")
    elif mode in {"return", "debug"}:
        return None
    elif mode == "default":
        return "01.01.1900"
    else:
        raise NotImplementedError(
            f"`mode` can only be one of {available_modes} (Case sensitive)"
        )
