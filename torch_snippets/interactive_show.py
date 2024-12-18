# AUTOGENERATED! DO NOT EDIT! File to edit: ../nbs/interactive_show.ipynb.

# %% auto 0
__all__ = ['COLORS', 'to_networkx', 'plot_image', 'plot_graph', 'tonp', 'tolist', 'convert_to_nx', 'viz2', 'df2graph_nodes',
           'ishow']

# %% ../nbs/interactive_show.ipynb 1
from pathlib import Path
from . import *
from .bokeh_loader import bshow
from bokeh.io import output_notebook, show as bokeh_show, output_file
from bokeh.plotting import figure, from_networkx
from bokeh.models import (
    Circle,
    Rect,
    WheelZoomTool,
    PanTool,
    BoxZoomTool,
    ResetTool,
    MultiLine,
    NodesAndLinkedEdges,
    EdgesAndLinkedNodes,
    HoverTool,
    TapTool,
    BoxSelectTool,
)
from bokeh.palettes import Spectral7
import networkx as nx

import numpy as np
from fastcore.basics import ifnone
from typing import Optional, Any, Iterable, Union
from .bb_utils import split_bb_to_xyXY, to_absolute, to_relative

output_notebook()

# %% ../nbs/interactive_show.ipynb 2
def to_networkx(
    data,
    node_attrs: Optional[Iterable[str]] = None,
    edge_attrs: Optional[Iterable[str]] = None,
    graph_attrs: Optional[Iterable[str]] = None,
    to_undirected: Optional[Union[bool, str]] = False,
    remove_self_loops: bool = False,
) -> Any:
    import torch

    r"""Converts a :class:`torch_geometric.data.Data` instance to a
    :obj:`networkx.Graph` if :attr:`to_undirected` is set to :obj:`True`, or
    a directed :obj:`networkx.DiGraph` otherwise.

    Args:
        data (torch_geometric.data.Data): The data object.
        node_attrs (iterable of str, optional): The node attributes to be
            copied. (default: :obj:`None`)
        edge_attrs (iterable of str, optional): The edge attributes to be
            copied. (default: :obj:`None`)
        graph_attrs (iterable of str, optional): The graph attributes to be
            copied. (default: :obj:`None`)
        to_undirected (bool or str, optional): If set to :obj:`True` or
            "upper", will return a :obj:`networkx.Graph` instead of a
            :obj:`networkx.DiGraph`. The undirected graph will correspond to
            the upper triangle of the corresponding adjacency matrix.
            Similarly, if set to "lower", the undirected graph will correspond
            to the lower triangle of the adjacency matrix. (default:
            :obj:`False`)
        remove_self_loops (bool, optional): If set to :obj:`True`, will not
            include self loops in the resulting graph. (default: :obj:`False`)

    Examples:

        >>> edge_index = torch.tensor([
        ...     [0, 1, 1, 2, 2, 3],
        ...     [1, 0, 2, 1, 3, 2],
        ... ])
        >>> data = Data(edge_index=edge_index, num_nodes=4)
        >>> to_networkx(data)
        <networkx.classes.digraph.DiGraph at 0x2713fdb40d0>

    """
    import networkx as nx

    G = nx.Graph() if to_undirected else nx.DiGraph()

    G.add_nodes_from(range(data.num_nodes))

    node_attrs = node_attrs or []
    edge_attrs = edge_attrs or []
    graph_attrs = graph_attrs or []

    values = {}
    for key, value in data(*(node_attrs + edge_attrs + graph_attrs)):
        if torch.is_tensor(value):
            value = value if value.dim() <= 1 else value.squeeze(-1)
            values[key] = value.tolist()
        else:
            values[key] = value

    to_undirected = "upper" if to_undirected is True else to_undirected
    to_undirected_upper = True if to_undirected == "upper" else False
    to_undirected_lower = True if to_undirected == "lower" else False

    for i, (u, v) in enumerate(data.edge_index.t().tolist()):

        if to_undirected_upper and u > v:
            continue
        elif to_undirected_lower and u < v:
            continue

        if remove_self_loops and u == v:
            continue

        G.add_edge(u, v)

        for key in edge_attrs:
            G[u][v][key] = values[key][i]

    for key in node_attrs:
        for i, feat_dict in G.nodes(data=True):
            feat_dict.update({key: values[key][i]})

    for key in graph_attrs:
        G.graph[key] = values[key]

    return G


def plot_image(p, image, sz):
    if isinstance(image, str):
        image = read(image, 1)
    imga = resize(image, sz)
    h, w = imga.shape[:2]
    img = np.empty((h, w), dtype=np.uint32)
    view = img.view(dtype=np.uint8).reshape((h, w, 4))
    for i in range(h):
        for j in range(w):
            view[i, j, :3] = imga[h - i - 1, j]
            view[i, j, 3] = 255

    p.x_range.range_padding = p.y_range.range_padding = 0
    p.image_rgba(image=[img], x=0, y=0, dw=1, dh=h / w)


COLORS = {v[0]: v for v in ["green", "red", "blue", "yellow", "cyan", "magenta"]}


def plot_graph(g, output, im=None, **kwargs):
    plot = figure(
        title=kwargs.get("title", "Networkx Integration Demonstration"),
        tools=[WheelZoomTool(), PanTool(), BoxZoomTool(), ResetTool()],
        width=900,
        height=900,
        match_aspect=True,
    )
    h, w = im.shape[:2]
    plot_image(plot, im, sz=kwargs.get("sz", 0.5))
    if nx.get_node_attributes(g, "color") != {}:
        nx.set_node_attributes(
            g,
            {ix: COLORS[c] for ix, c in nx.get_node_attributes(g, "color").items()},
            "node_color",
        )
    else:
        nx.get_node_attributes(g, "pos").items()
        fill_color = {ix: "green" for ix, _ in nx.get_node_attributes(g, "pos").items()}
        nx.set_node_attributes(g, fill_color, "node_color")
    hover_tool = HoverTool(
        tooltips=[("index", "$index")]
        + [(i, f"@{i}") for i in kwargs.get("tooltips", [])]
    )

    plot.add_tools(
        hover_tool,
        TapTool(),
        BoxSelectTool(),
    )
    plot.toolbar.active_scroll = plot.select_one(WheelZoomTool)
    plot.toolbar.active_drag = plot.select_one(PanTool)

    if kwargs.get("pos"):
        pos = nx.get_node_attributes(g, kwargs.get("pos"))
        pos = {i: (x, y * h / w) for i, (x, y) in pos.items()}
        graph = from_networkx(g, pos)
    else:
        print("Using random")
        graph = from_networkx(g, nx.random_layout)

    graph.node_renderer.glyph = Rect(
        width="w",
        height="h",
        # fill_color=kwargs.get("color", "green"),
        fill_color="node_color",
        fill_alpha=kwargs.get("opacity", 0.3),
    )
    plot.renderers.append(graph)
    bshow(plot)


def tonp(i):
    import torch

    if isinstance(i, torch.Tensor):
        i = i.cpu().detach().numpy()
        return i
    if isinstance(i, list):
        return np.array(i)


def tolist(i):
    import torch

    if isinstance(i, torch.Tensor):
        i = tonp(i)
    if isinstance(i, np.ndarray):
        i = i.tolist()
    return i


def convert_to_nx(g, node_attrs=None, undirected=True):
    graph = to_networkx(g, to_undirected=undirected)
    if node_attrs:
        for attr in node_attrs:
            _attr = tolist(g[attr])
            nx.set_node_attributes(
                graph, {i: _attr[i] for i in range(graph.number_of_nodes())}, attr
            )
    return graph


def viz2(graph, node_attrs=None, undirected=True, **kwargs):
    if not isinstance(graph, nx.classes.graph.Graph):
        graph = convert_to_nx(graph, node_attrs=node_attrs, undirected=undirected)
    plot_graph(graph, "networkx_graph", **kwargs)


def df2graph_nodes(
    df,
    text_attr="text",
    additional_attrs=None,
):
    df = df.copy()
    bbs = df2bbs(df)
    pos = [(bb.xc, 1 - bb.yc) for bb in bbs]
    w = [bb.w for bb in bbs]
    h = [bb.h for bb in bbs]
    if additional_attrs:
        df = df[additional_attrs]
    df["pos"] = pos
    df["w"] = w
    df["h"] = h
    nodes = [(ix, node) for ix, node in enumerate(df.to_dict(orient="records"))]
    return nodes


def ishow(im, df, additional_attrs=None, **kwargs):
    if isinstance(df, (str, Path)):
        df = str(df)
        df = pd.read_csv(df) if df.endswith("csv") else pd.read_parquet(df)
    df = df.copy()
    if isinstance(im, (str, P)):
        im = read(im, 1)
    elif isinstance(im, PIL.Image.Image):
        im = np.array(im)
    h, w = im.shape[:2]
    df = df.reset_index(drop=True)
    G = nx.Graph()
    df = split_bb_to_xyXY(df)
    df = to_relative(df, h, w)
    if additional_attrs is None:
        additional_attrs = [c for c in df.columns if c not in [*"xyXY"]]
    G.add_nodes_from(df2graph_nodes(df, additional_attrs=additional_attrs))
    viz2(G, im=im, tooltips=additional_attrs, pos="pos", **kwargs)
