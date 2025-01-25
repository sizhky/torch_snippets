__all__ = ["make_uniform_date_format", "ALL_DATE_FORMATS", "are_dates_equal", "today"]

from datetime import datetime
from torch_snippets.loader import flatten, Debug
from itertools import product

seps = list(",.-/ ")
seps = (
    seps
    + [f" {s} " for s in seps if s != " "]
    + [f"{s} " for s in seps if s != " "]
    + [f" {s}" for s in seps if s != " "]
)

x = flatten(
    [
        [
            "%d{s1}%m{s2}%Y".format(s1=s1, s2=s2),
            "%Y{s1}%m{s2}%d".format(s1=s1, s2=s2),
            "%d{s1}%b{s2}%Y".format(s1=s1, s2=s2),
            "%b{s1}%d{s2}%Y".format(s1=s1, s2=s2),
            "%m{s1}%d{s2}%Y".format(s1=s1, s2=s2),
        ]
        for s1, s2 in product(seps, repeat=2)
    ]
)
x = x + [_x.replace("%b", "%B") for _x in x]
x = x + [_x.replace("%Y", "%y") for _x in x]
x = x + [_x.replace("%d", "%-d") for _x in x]
x = x + [_x.replace("%m", "%-m") for _x in x]

ALL_DATE_FORMATS = x + ["%Y-%m-%d %H:%M:%S"]


def make_uniform_date_format(value, target_fmt="%d.%m.%Y", mode="raise", debug=False):
    available_modes = ["raise", "return", "default"]
    if isinstance(value, datetime):
        return value.strftime(target_fmt)
    for fmt in ALL_DATE_FORMATS:
        try:
            output = datetime.strptime(value, fmt).strftime(target_fmt)
            # Debug(f"{value=}, {output=}, {fmt=}")
            if debug:
                return output, fmt
            return output
        except:
            ...
    if mode == "raise":
        raise NotImplementedError(f"Unable to give a proper date for `{value}`")
    elif mode in {"return"}:
        return None
    elif mode == "default":
        return "01.01.1900"
    else:
        raise NotImplementedError(
            f"`mode` can only be one of {available_modes} (Case sensitive)"
        )


def are_dates_equal(date1, date2):
    try:
        date1 = make_uniform_date_format(date1)
        date2 = make_uniform_date_format(date2)

        return date1 == date2
    except:
        return False


def today(fmt="%Y%m%d"):
    return datetime.today().strftime(fmt)
