"""Interactive HTML view of the subgraph an answer used (pyvis, JS inlined for offline demos)."""
from pyvis.network import Network

COLORS = {"Drug": "#4e79a7", "Enzyme": "#f28e2b", "DrugClass": "#59a14f",
          "Condition": "#e15759", "SideEffect": "#b07aa1", "Unknown": "#9c9c9c"}


def subgraph_html(node_types: dict[str, str], path_nodes: list[str], path_edges: list[dict],
                  entities: list[str], height: str = "520px") -> str:
    net = Network(height=height, width="100%", directed=True, bgcolor="#ffffff",
                  font_color="#222222", cdn_resources="in_line")
    nodes = set(path_nodes) | {e["source"] for e in path_edges} | {e["target"] for e in path_edges}
    for n in sorted(nodes):
        t = node_types.get(n, "Unknown")
        is_query = n in entities
        net.add_node(n, label=n, title=t, color=COLORS.get(t, COLORS["Unknown"]),
                     size=30 if is_query else 16, borderWidth=4 if is_query else 1)
    seen = set()
    for e in path_edges:
        k = (e["source"], e["target"], e["type"])
        if k in seen:
            continue
        seen.add(k)
        net.add_edge(e["source"], e["target"], label=e["type"], title=e.get("evidence", ""), width=2)
    net.set_options('{"physics": {"stabilization": {"iterations": 150}},'
                    ' "edges": {"font": {"size": 11, "align": "middle"}, "arrows": {"to": {"enabled": true}}}}')
    return net.generate_html()
