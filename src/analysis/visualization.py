import json

from networkx import DiGraph, nx_agraph, spring_layout
from process_execution import ProcessExecution

def apply_node_styles_nx(
    G: ProcessExecution, type_attr="type", nested_attr="attr", color_map=None, default_color="#d3d3d3"
):
    """
    Compute style attributes for nodes from a NetworkX graph G and store them
    as top-level node attributes so they are preserved after nx -> AGraph conversion.

    Sets: 'style' (filled), 'fillcolor' and 'tooltip' (string).
    - G: networkx.Graph or DiGraph with node data dicts
    - type_attr: key (in the node data dict or inside data['attr']) holding node type
    - nested_attr: key that may contain a dict/stringified dict with more attributes
    - color_map: dict mapping type -> color
    - default_color: fallback color
    """
    if color_map is None:
        color_map = {
            "OBJECT": "#085725",
            "EVENT": "#009ee1",
            "ACTIVITY": "#f8ed5d",
            "DEFAULT": default_color,
        }

    for node, data in G.nodes(data=True):
        tooltip_data = {}

        # First try direct top-level type
        ntype = data.get(type_attr)

        # If nested_attr exists, try to parse it (dict or JSON / Python literal)
        nested = data.get(nested_attr)
        if nested is not None:
            if isinstance(nested, dict):
                # prefer nested type if top-level missing
                if not ntype:
                    ntype = nested.get("type") or nested.get("ocel:type")
                tooltip_data = nested

        # If still no tooltip_data, collect visible node-level attrs
        if not tooltip_data:
            # copy node data but keep only simple serializable values
            tooltip_data = {
                k: v
                for k, v in data.items()
                if k not in ("style", "fillcolor", "tooltip")
            }

        # choose color
        color = color_map.get(ntype, color_map.get("DEFAULT", default_color))

        # build tooltip string (escape double quotes)
        tooltip_items = []
        for k, v in tooltip_data.items():
            try:
                s = str(v)
            except Exception:
                s = repr(v)
            s = s.replace('"', "'")
            tooltip_items.append(f"{k}={s}")
        tooltip_str = "\n".join(tooltip_items)

        # store style attributes on the networkx node (ensure strings)
        data["style"] = "filled"
        data["fillcolor"] = str(color)
        data["tooltip"] = tooltip_str

    return G

def apply_edge_styles_nx(
    G: ProcessExecution
):
    for u, v, ed in G.edges(data=True):
        if ed["attr"].get("type", "") == "DF":
            ed["style"] = "solid"
        else:
            ed["style"] = "dashed"
            ed["penwidth"] = "0.5"


def visualize_with_svg_and_js(trace_graph: ProcessExecution, normalized_subgraphs: list, out_prefix: str):
    """Create an interactive HTML using native SVG with Graphviz layout, preserving original agraph styles.
    Includes JavaScript-based toggle buttons to highlight normalized process executions.

    - trace_graph: full networkx DiGraph with node attributes set by apply_node_styles_nx
    - normalized_subgraphs: list of networkx subgraphs to be highlighted
    - out_prefix: output file prefix (files placed into `figures/`)
    """
    
    G = trace_graph
    
    # Render base graph as SVG using Graphviz
    agraph = nx_agraph.to_agraph(G)
    svg_str = agraph.draw(format='svg', prog='dot').decode('utf-8')
    
    # Escape svg_str for embedding in JavaScript string
    svg_str_escaped = svg_str.replace('`', '\\`').replace('\\', '\\\\')
    
    # Build highlight configs as JSON
    highlight_configs = []
    for subg in normalized_subgraphs:
        highlight_configs.append({
            'nodes': list(subg.nodes()),
            'edges': [(str(u), str(v)) for u, v in subg.edges()]
        })
    
    highlight_configs_json = json.dumps(highlight_configs)
    
    # Create HTML with embedded SVG + JavaScript toggle logic
    html_content = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>Process Execution - Interactive ({out_prefix})</title>
    <style>
        body {{
            font-family: Arial, sans-serif;
            margin: 20px;
            background-color: #f9f9f9;
        }}
        .container {{
            margin: 0 auto;
            background-color: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        }}
        h1 {{
            margin-top: 0;
            color: #333;
        }}
        .controls {{
            margin-bottom: 20px;
            display: flex;
            gap: 10px;
            flex-wrap: wrap;
        }}
        button {{
            padding: 8px 16px;
            font-size: 14px;
            border: 1px solid #ccc;
            border-radius: 4px;
            background-color: #f0f0f0;
            cursor: pointer;
            transition: background-color 0.2s;
        }}
        button:hover {{
            background-color: #e0e0e0;
        }}
        button.active {{
            background-color: #4CAF50;
            color: white;
            border-color: #4CAF50;
        }}
        .svg-container {{
            border: 1px solid #ddd;
            border-radius: 4px;
            overflow: auto;
            background-color: #fafafa;
            display: flex;
        }}
        svg {{
            display: block;
        }}
        .info {{
            margin-top: 15px;
            font-size: 12px;
            color: #666;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>Process Execution - Interactive</h1>
        <div class="controls" id="controls"></div>
        <div class="svg-container" id="svg-container"></div>
        <div class="info">
            Click buttons to toggle highlights of normalized process executions.
        </div>
    </div>
    
    <script>
        const highlightConfigs = {highlight_configs_json};
        const svgStr = `{svg_str_escaped}`;
        const svgContainer = document.getElementById('svg-container');
        svgContainer.innerHTML = svgStr;
        
        const controls = document.getElementById('controls');
        let currentHighlight = null;
        
        // Create 'None' button
        const noneBtn = document.createElement('button');
        noneBtn.textContent = 'None';
        noneBtn.className = 'active';
        noneBtn.onclick = () => clearHighlight();
        controls.appendChild(noneBtn);
        
        // Create highlight buttons
        highlightConfigs.forEach((config, idx) => {{
            const btn = document.createElement('button');
            btn.textContent = `Highlight ${{idx + 1}}`;
            btn.onclick = () => applyHighlight(idx, btn);
            controls.appendChild(btn);
        }});
        
        function clearHighlight() {{
            // Reset all node/edge styles back to original
            document.querySelectorAll('ellipse, polygon, polyline, path').forEach(el => {{
                const orig_stroke = el.getAttribute('data-orig-stroke');
                const orig_stroke_width = el.getAttribute('data-orig-stroke-width');
                if (orig_stroke !== null) el.style.stroke = orig_stroke;
                if (orig_stroke_width !== null) el.style.strokeWidth = orig_stroke_width;
            }});
            
            document.querySelectorAll('button').forEach(btn => {{
                btn.classList.remove('active');
            }});
            document.querySelector('button').classList.add('active');
            currentHighlight = null;
        }}
        
        function applyHighlight(idx, btn) {{
            // Save original styles if not already saved
            document.querySelectorAll('ellipse, polygon, polyline, path').forEach(el => {{
                if (!el.hasAttribute('data-orig-stroke')) {{
                    el.setAttribute('data-orig-stroke', el.style.stroke || el.getAttribute('stroke') || '');
                    el.setAttribute('data-orig-stroke-width', el.style.strokeWidth || el.getAttribute('stroke-width') || '');
                }}
            }});
            
            clearHighlight();
            btn.classList.add('active');
            const config = highlightConfigs[idx];
            
            // Highlight nodes with orange border
            config.nodes.forEach(nodeId => {{
                const nodeStr = String(nodeId);
                document.querySelectorAll('g[class="node"]').forEach(elem => {{
                    const title = elem.querySelector('title');
                    const ellipse = elem.querySelector('ellipse');
                    if (title && title.textContent.includes(nodeStr)) {{
                        ellipse.style.stroke = '#ff7f0e';
                        ellipse.style.strokeWidth = '3';
                    }}
                }});
            }});
            
            // Highlight edges with orange color
            config.edges.forEach(([u, v]) => {{
                const uStr = String(u);
                const vStr = String(v);
                document.querySelectorAll('g[class="edge"]').forEach(elem => {{
                    const title = elem.querySelector('title');
                    if (title) {{
                        const text = title.textContent;
                        if ((text.includes(uStr) && text.includes(vStr)) || 
                            (text.includes(vStr) && text.includes(uStr))) {{
                            elem.querySelectorAll('polyline, polygon, path').forEach(drawing => {{
                                drawing.style.stroke = '#ff7f0e';
                                drawing.style.strokeWidth = '2';
                            }})
                        }}
                    }}
                }});
            }});
            
            currentHighlight = idx;
        }}
    </script>
</body>
</html>
"""
    
    # Write HTML file
    html_path = f"figures/{out_prefix}_interactive.html"
    with open(html_path, 'w') as f:
        f.write(html_content)
    
    return html_path


def visualize_highlight_normalized(trace_graph: ProcessExecution, normalized_subgraphs: list, out_prefix: str):
    # Additionally, produce small highlighted SVGs for each normalized subgraph (optional)
    out_paths = []
    for idx, subg in enumerate(normalized_subgraphs):
        try:
            # Create a copy of the full graph to modify styles for highlighting
            H = DiGraph(trace_graph)

            # default dim style for nodes/edges
            dim_node_color = "#e6e6e6"
            dim_edge_color = "#bdbdbd"
            highlight_node_color = "#ff7f0e"  # orange
            highlight_edge_color = "#ff7f0e"

            # set dim styles first
            for n, d in H.nodes(data=True):
                d["style"] = "filled"
                d["fillcolor"] = dim_node_color
            for u, v, ed in H.edges(data=True):
                ed["color"] = dim_edge_color
                ed["penwidth"] = "1"

            # highlight nodes in the normalized subgraph
            sub_nodes = set(subg.nodes())
            for n in sub_nodes:
                if n in H.nodes():
                    H.nodes[n]["fillcolor"] = highlight_node_color
                    H.nodes[n]["style"] = "filled"

            # highlight edges in the normalized subgraph
            for u, v in subg.edges():
                if H.has_edge(u, v):
                    H.edges[u, v]["color"] = highlight_edge_color
                    H.edges[u, v]["penwidth"] = "2"

            # render highlighted version
            agraph_h = nx_agraph.to_agraph(H)
            out_path = f"figures/{out_prefix}_normalized_{idx}.svg"
            agraph_h.draw(out_path, prog="dot")
            out_paths.append(out_path)
        except Exception:
            # non-fatal; continue to next
            continue

    return out_paths
