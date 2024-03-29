__all__ = [
    "textify",
    "find_lines",
    "find_blocks",
    "find_substring",
    "get_line_data_from_word_data",
    "edit_distance_path",
    "group_blocks",
]
import string


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from fuzzywuzzy import fuzz
from sklearn.cluster import dbscan

from torch_snippets import *


def textify(dataframe: pd.DataFrame, im=None, separator_size: int = 20):
    """Create text out of a dataframe returned by im2df"""
    assert isinstance(dataframe, pd.DataFrame)
    if dataframe is None:
        return "", 0
    show = True if im is not None else False
    dataframe = left_to_right_top_to_bottom(dataframe)
    bbs = df2bbs(dataframe)
    if show:
        fig, ax = plt.subplots(1, figsize=(20, 20))
        ax.imshow(im, cmap="gray")
    TEXT = ""
    prev_top = 0
    prev_left = 0
    for ix, row in dataframe.iterrows():
        x, y, X, Y = bb = bbs[ix]
        l, t = x, y
        w, h = bb.w, bb.h
        curr_top = t
        curr_left = l
        sep = (
            "\n"
            if (curr_top - prev_top > separator_size) or (curr_left - prev_left > 500)
            else " "
        )
        text = row["text"]
        text = "" if text is None else text
        TEXT = TEXT + sep + text
        prev_top = t
        prev_left = l
        if show:
            ax.text(l, t - 20, "<{}>".format(row["text"]))
            # TODO: What is patches?
            rect = None
            # rect = patches.Rectangle((l, t), w, h, angle=0.0, linewidth=np.random.randint(2, 5),
            #                          edgecolor=np.random.rand(3), facecolor='none')
            ax.add_patch(rect)
    if show:
        plt.show()
    return TEXT


def find_lines(df=None, eps=20):
    df = df.reset_index()
    if "line" in df.columns:
        df.drop("line", inplace=True, axis=1)
    bbs = df2bbs(df)
    extra_columns = df.columns.tolist()
    for col in ["bb", "x", "y", "X", "Y"]:
        if col in extra_columns:
            extra_columns.remove(col)
    extra_columns = {col: df[col] for col in extra_columns}

    df = find_lines_for_bbs(bbs, eps=eps)
    for col in extra_columns:
        df = pd.merge(df, extra_columns[col], left_on="i", right_index=True)
    df.drop(["i"], inplace=True, axis=1)
    df.columns = [
        *"xyXY",
        "line",
        *extra_columns.keys(),
    ]  # "x,y,X,Y,line,text,conf,type".split(",")
    return df


def find_blocks(df, eps=20):
    blocks = []
    for line in sorted(df["line"].unique()):
        _df = df[df["line"] == line].sort_values("x")
        if line % 1 != 0:
            _df["line_block"] = line
        else:
            _df["delta_x"] = _df["x"] - _df["X"].shift(1)
            _df["line_block"] = (
                df["line"].map(str)
                + "_"
                + ((_df["x"] - _df["X"].shift(1)) > eps).cumsum().map(str)
            )
        blocks.append(_df)
    return pd.concat(blocks)


def find_substring(needle, hay):
    hay = hay.replace("\n", " ")
    needle_length = len(needle.split())
    max_sim_val = 0
    max_sim_string = ""
    for ix, ngram in enumerate(ngrams(hay.split(), needle_length)):
        hay_ngram = " ".join(ngram)
        similarity = SM(None, hay_ngram, needle).ratio()
        if similarity > max_sim_val:
            max_sim_val = similarity
            max_sim_string = hay_ngram

    for ix, ngram in enumerate(
        ngrams(hay.split(), needle_length + int(0.25 * needle_length))
    ):
        hay_ngram = " ".join(ngram)
        similarity = SM(None, hay_ngram, needle).ratio()
        if similarity > max_sim_val:
            max_sim_val = similarity
            max_sim_string = hay_ngram

    for ix, ngram in enumerate(
        ngrams(hay.split(), needle_length + int(0.5 * needle_length))
    ):
        hay_ngram = " ".join(ngram)
        similarity = SM(None, hay_ngram, needle).ratio()
        if similarity > max_sim_val:
            max_sim_val = similarity
            max_sim_string = hay_ngram

    for ix, ngram in enumerate(
        ngrams(hay.split(), needle_length + int(0.75 * needle_length))
    ):
        hay_ngram = " ".join(ngram)
        similarity = SM(None, hay_ngram, needle).ratio()
        if similarity > max_sim_val:
            max_sim_val = similarity
            max_sim_string = hay_ngram

    if max_sim_string.strip() == "":
        return "", 0, [hay]
    rest_of_strings = hay.split(max_sim_string)
    rest_of_strings = [s.strip() for s in rest_of_strings if s.strip() != ""]
    return max_sim_string, max_sim_val, rest_of_strings


def find_lines_for_bbs(bbs, eps=20):
    """Find Lines For Identified Bounding Boxes"""
    if len(bbs) == 0:
        return pd.DataFrame(columns=list("xyXY"))
    df = pd.DataFrame([(ix, x, y, X, Y) for ix, (x, y, X, Y) in enumerate(bbs)])
    df.columns = list("ixyXY")
    ys = []
    for bb in bbs:
        ys.append((bb[1], bb[3]))
    _ys = np.array(ys)
    db = dbscan(_ys, eps, min_samples=2)
    lines = np.c_[_ys, db[1]]
    # Applied dbscan to the y,Y values
    df["line_"] = db[1]
    df_ = df.groupby("line_").agg({"Y": "mean"})
    df_ = df_.sort_values("Y", ascending=True)
    for index in df_.index:
        if index == -1:
            df_.drop([-1], axis=0, inplace=True)
    # Created a dataframe that contains mean Y value for each row
    line__ = {}
    lctr = 0
    for line_ in df_.index:
        line__[line_] = lctr
        lctr += 1
    line__[-1] = -1
    # Created a dictionary to check if the numbering goes wrong
    df["line"] = df.line_.map(lambda x: float(line__[x]))
    df_.index = df_.index.map(lambda x: line__[x])
    # Mapped the row numbers with the dictionary numbers
    for row in range(len(df)):
        if df.loc[row, "line_"] != -1:
            continue
        tmp = df_[df_["Y"] < df.loc[row, "Y"]]
        if tmp.index.max() is np.nan:
            df.loc[row, "line"] = -0.5
        else:
            df.loc[row, "line"] = tmp.index.max() + 0.5
    df = df.drop(columns=["line_"])
    df.sort_values(["line", "X"], inplace=True)
    df = df.reset_index(drop=True)
    df1 = df[df["line"] % 1 == 0.5].reset_index()
    df2 = df[df["line"] % 1 != 0.5]
    idx = list(df1["index"])
    df1 = df1.drop(columns=["index"]).sort_values(["Y"])
    df1["index"] = idx
    df2 = df2.reset_index()
    df = pd.concat([df2, df1], axis=0, sort=False)
    df.sort_values(["index"], inplace=True)
    df = df.drop(columns=["index"]).reset_index(drop=True)
    better_lines(df)
    return df


def better_lines(df):
    for line in df[df.line % 1 == 0.5].line.unique():
        _x = df[df["line"] == line]
        s, e = line - 0.5, line + 0.5
        new_lines = np.linspace(s, e, len(_x) + 2)[1:-1]
        df.loc[_x.index, "line"] = new_lines


def divide_text_lines(df, eps=10):
    if len(df) < 1:
        return df
    elif len(df) == 1:
        df["line"] = 0
    else:
        line = [None] * len(df)
        line_count = 0
        for r_idx, row in df.iterrows():
            if r_idx == 0:
                text, conf, x, y, X, Y = row
                line[r_idx] = line_count
            else:
                text1, conf1, x1, y1, X1, Y1 = row
                if abs(y - y1) < eps:
                    line[r_idx] = line_count
                else:
                    line_count += 1
                    line[r_idx] = line_count
                    text, conf, x, y, X, Y = row
        df["line"] = line
    df = df.sort_values(["line", "x"]).reset_index(drop=True)
    return df


def left_to_right_top_to_bottom(df):
    if "line" in [c.lower() for c in df.columns]:
        return df.sort_values(["line", "x"])
    else:
        return df.sort_values(["y", "x"])


def im2phrases(df, im=None, seperator_size: int = 20):
    if len(df) == 0:
        return "", 0
    show = False if im is None else True
    if show:
        fig, ax = plt.subplots(1, figsize=(20, 20))
        ax.imshow(im, cmap="gray")
    TEXT, text = [], []
    POINTS, points = [], []
    CONF, conf = [], []
    for r_idx, row in df.iterrows():
        x, y, X, Y = row["x"], row["y"], row["X"], row["Y"]
        w, h = X - x, Y - y
        if r_idx == 0:
            prev_l, prev_t, prev_r, prev_b, prev_line, prev_text, prev_conf = (
                x,
                y,
                X,
                Y,
                row["line"],
                asciify(row["text"]),
                row["conf"],
            )
            points.append((prev_l, prev_t, prev_r, prev_b))
            text.append(prev_text)
            conf.append(prev_conf)
            continue
        else:
            curr_l, curr_t, curr_r, curr_b, curr_line, curr_text, curr_conf = (
                x,
                y,
                X,
                Y,
                row["line"],
                asciify(row["text"]),
                row["conf"],
            )

        if (curr_line != prev_line) or abs(prev_r - curr_l) > 20:
            TEXT.append(" ".join(text))
            POINTS.append(points)
            CONF.append(mean(conf))
            text, points, conf = [], [], []
            text.append(curr_text)
            points.append((curr_l, curr_t, curr_r, curr_b))
            conf.append(curr_conf)
        else:
            text.append(curr_text)
            points.append((curr_l, curr_t, curr_r, curr_b))
            conf.append(curr_conf)

        prev_l, prev_t, prev_r, prev_b, prev_line = (
            curr_l,
            curr_t,
            curr_r,
            curr_b,
            curr_line,
        )
    if r_idx == len(df) - 1:
        TEXT.append(" ".join(text))
        POINTS.append(points)
        CONF.append(mean(conf))
    df = pd.DataFrame(
        data=list(zip(TEXT, POINTS, CONF)),
        columns=["text_lines", "line_ends(x,y,X,Y)", "conf"],
    )

    return df


def asciify(string):
    return "".join([c for c in string if c in valid_chars])


def find_match(word, words, N=1, lower=False):
    if word is None:
        return None, None
    if lower:
        word = word.lower()
        words = [w.lower() for w in words]
    matches, lengths = [], []
    for ix, f in enumerate(words):
        # if len(word)<=len(f)//2 or len(f)<=len(word)//2:
        # 	matches.append(-1)
        # 	lengths.append(-1)
        # 	continue
        k = min(len(f), len(word)) + 2
        m = fuzz.ratio(word[:k], f[:k])
        matches.append(m)
        lengths.append(k)
    sorted_idx = np.argsort(matches)
    if N > 1:
        return sorted_idx[:N], [words[ix] for ix in sorted_idx[:N]]
    else:
        M = max(matches)
        ixs = [ix for ix, m in enumerate(matches) if m == M]
        if len(ixs) == 1:
            return np.argmax(matches), max(matches)
        else:
            ix = max(ixs, key=lambda x: lengths[x])
            ix = min(ixs)
            return ix, matches[ix]


from difflib import SequenceMatcher as SM


def fuzzyfind(needle, hay):
    needle_length = len(needle.split())
    max_sim_val = 0
    max_sim_string = ""

    for ngram in ngrams(hay.split(), needle_length + int(0.2 * needle_length)):
        hay_ngram = " ".join(ngram)

        # TODO: No reference to similarity
        # if similarity > max_sim_val:
        #     max_sim_val = similarity
        #     max_sim_string = hay_ngram

    return max_sim_val, max_sim_string, hay.find(max_sim_string)


valid_chars = string.ascii_letters + "1234567890 !@#$%^&*(){}[];':\",./<>?-_=+"


def get_line_data_from_word_data(df, line_gap=10, block_gap=10):
    df = find_lines(df, eps=line_gap)
    df = find_blocks(df, eps=block_gap)
    df = df.groupby("line_block").agg(
        {
            "x": "min",
            "y": "min",
            "X": "max",
            "Y": "max",
            "text": lambda x: " ".join(x),
            **{c: lambda x: list(x) for c in df.columns if c not in [*"xyXY", "text"]},
        }
    )
    return df


def edit_distance_path(string1, string2):
    if string1 is None or string2 is None:
        return -1, []
    m = len(string1)
    n = len(string2)

    # Create a matrix to store the edit distances
    distance_matrix = [[0] * (n + 1) for _ in range(m + 1)]

    # Initialize the first row and column
    for i in range(m + 1):
        distance_matrix[i][0] = i
    for j in range(n + 1):
        distance_matrix[0][j] = j

    # Calculate the edit distances
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if string1[i - 1] == string2[j - 1]:
                distance_matrix[i][j] = distance_matrix[i - 1][j - 1]
            else:
                distance_matrix[i][j] = min(
                    distance_matrix[i - 1][j] + 1,  # deletion
                    distance_matrix[i][j - 1] + 1,  # insertion
                    distance_matrix[i - 1][j - 1] + 1,  # substitution
                )

    # Trace back the edit distance path
    i = m
    j = n
    path = []

    while i > 0 and j > 0:
        if string1[i - 1] == string2[j - 1]:
            path.append((string1[i - 1], string2[j - 1], ""))
            i -= 1
            j -= 1
        else:
            if distance_matrix[i][j] == distance_matrix[i - 1][j] + 1:
                path.append((string1[i - 1], "", "delete"))
                i -= 1
            elif distance_matrix[i][j] == distance_matrix[i][j - 1] + 1:
                path.append(("", string2[j - 1], "insert"))
                j -= 1
            else:
                path.append((string1[i - 1], string2[j - 1], "replace"))
                i -= 1
                j -= 1

    while i > 0:
        path.append((string1[i - 1], "", "delete"))
        i -= 1

    while j > 0:
        path.append(("", string2[j - 1], "insert"))
        j -= 1

    # Reverse the path to get the correct order
    path.reverse()

    return distance_matrix[m][n], path


def group_blocks(df):
    output = []
    for block, _df in df.groupby("line_block"):
        texts = [t for t in _df.text if isinstance(t, str)]
        _block = {
            "text": " ".join(texts),
            "x": _df.x.min(),
            "y": _df.y.min(),
            "X": _df.X.max(),
            "Y": _df.Y.max(),
            "line_block": block,
        }
        output.append(_block)
    output = pd.DataFrame(output)
    return output
