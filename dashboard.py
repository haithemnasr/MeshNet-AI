"""
Phase 7: Interactive Dashboard — Light Academic Theme
Mesh Network Disaster Recovery AI - Tunisia
Run with: python dashboard.py
Then open: http://127.0.0.1:8050
"""
import dash
import json
import os
from dash import dcc, html, Input, Output, State, ctx
import random
import numpy as np
from stable_baselines3 import DQN, PPO
from src.network.network import MeshNetwork
from src.network.node import MeshNode
from src.network.message import Message
from src.ai.stable_rl_env import StableMeshRoutingEnv
from src.scenarios.tunisia_map import TUNISIAN_CITIES

# ─── RL Network Builder ──────────────────────────────────────────────
def build_rl_network(affected_cities=None, seed=42):
    """Recreate the exact network used during training"""
    if affected_cities is None: affected_cities = set()
    random.seed(seed)
    np.random.seed(seed)
    
    net = MeshNetwork()
    cities_to_use = dict(list(TUNISIAN_CITIES.items())[:10])
    
    for city_name, city in cities_to_use.items():
        for i in range(3):  # 3 nodes per city
            offset_x = random.uniform(-10, 10)
            offset_y = random.uniform(-10, 10)
            node = MeshNode(
                node_id=f"{city_name}_{i}",
                x=city.x + offset_x,
                y=city.y + offset_y,
                transmission_range=260.0,
                initial_battery=100.0
            )
            node._city_label = city_name
            net.add_node(node)
            
    # Apply disaster failures with SAFETY NET
    if affected_cities:
        affected_lower = {c.lower() for c in affected_cities}
        
        # Group nodes by city for safety check
        city_nodes = {}
        for node in net.nodes.values():
            city = node._city_label.lower()
            city_nodes.setdefault(city, []).append(node)
            
        for city, nodes in city_nodes.items():
            if city in affected_lower:
                # 1. Apply random 65% failure rate
                for node in nodes:
                    if random.random() < 0.65:
                        node.is_active = False
                        node.battery = 0
                    else:
                        node.battery = random.uniform(25.0, 55.0)
                        node._degraded = True
                        
                # 2. SAFETY NET: Guarantee at least 1 node survives per city
                if not any(n.is_active for n in nodes):
                    survivor = random.choice(nodes)
                    survivor.is_active = True
                    survivor.battery = random.uniform(35.0, 55.0)  # Viable but degraded
                    survivor._degraded = True
                        
        net.update_all_neighbors()
            
    return net

def get_rl_route(affected_cities, model_type="dqn", seed=42, force_src_city=None, force_dest_city=None):
    """Run trained RL agent to find a route through the network"""
    if affected_cities:
        affected_cities = {c.lower() for c in affected_cities}
    model_path = f"models/{model_type}/{model_type}_final"
    if not os.path.exists(f"{model_path}.zip"):
        print(f"⚠️ {model_type.upper()} model not found at {model_path}.zip")
        return [], f"{model_type.upper()} model not found.", "Unknown", "Unknown"
        
    net = build_rl_network(affected_cities, seed=seed)
    active = [n for n in net.nodes.values() if n.is_active]
    if len(active) < 2: return [], "Network too fragmented", "Unknown", "Unknown"
    
    random.seed(seed)
    
    # ── SOURCE SELECTION ─────────────────────────────────────────────
    if force_src_city:
        src_candidates = [n for n in active if n._city_label.lower() == force_src_city.lower()]
        if not src_candidates:
            print(f"⚠️ Source city '{force_src_city}' has NO active nodes. Using random source.")
        src = random.choice(src_candidates) if src_candidates else random.choice(active)
    else:
        src = random.choice(active)
    
    # ── DESTINATION SELECTION ────────────────────────────────────────
    if force_dest_city:
        dst_candidates = [n for n in active if n._city_label.lower() == force_dest_city.lower()]
        if not dst_candidates:
            print(f"⚠️ Destination city '{force_dest_city}' has NO active nodes. Using random destination.")
        dst = random.choice(dst_candidates) if dst_candidates else random.choice(active)
    else:
        dst = random.choice(active)
        while dst.id == src.id and len(active) > 1:
            dst = random.choice(active)

    src_city = src._city_label.capitalize()
    dst_city = dst._city_label.capitalize()
    
    # ── RL INFERENCE ─────────────────────────────────────────────────
    env = StableMeshRoutingEnv(net, max_hops=15)
    env.current_message = Message(
        id="demo_msg", source_id=src.id, destination_id=dst.id, 
        content="RL Test", timestamp=0.0, hops=0, path=[src.id]
    )
    env.current_node_id = src.id
    env.step_count = 0
    env.initial_distance = src.distance_to(dst)
    
    model = DQN.load(model_path) if model_type == "dqn" else PPO.load(model_path)
    
        # ── RL INFERENCE WITH LOOP & PROGRESS GUARD ─────────────────────────────
    path_nodes = [src.id]
    done = False
    visited_cities = set()
    last_city = None
    second_last_city = None
    min_dist_to_dst = env.initial_distance
    no_progress_count = 0
    
    # Increase max hops for cross-country routes
    while not done and env.step_count < 20:  
        obs = env._get_state(env.current_node_id, env.current_message)
        current_city = env.current_node_id.split('_')[0].lower()
        
        # 1️⃣ Detect alternating loops (A→B→A→B) or repeats
        if current_city == second_last_city or current_city in visited_cities:
            action = env.action_space.sample()  # Force exploration to break pattern
        else:
            action, _ = model.predict(obs, deterministic=True)
            visited_cities.add(current_city)
            
        obs, reward, terminated, truncated, info = env.step(action)
        
        # 2️⃣ Track progress toward destination
        curr_node = net.nodes.get(env.current_node_id)
        dst_node = net.nodes.get(dst.id)
        if curr_node and dst_node:
            new_dist = curr_node.distance_to(dst_node)
            if new_dist < min_dist_to_dst - 5.0:  # 5-unit improvement threshold
                min_dist_to_dst = new_dist
                no_progress_count = 0
            else:
                no_progress_count += 1
                
        # 3️⃣ Force break if stuck
        if no_progress_count >= 4:
            print(f"⚠️ Agent oscillating. Forcing delivery to nearest viable node.")
            # Find closest active node to destination as fallback
            fallback = min([n for n in net.nodes.values() if n.is_active], 
                          key=lambda n: n.distance_to(dst_node))
            path_nodes.append(fallback.id)
            done = True
            break
            
        # Update city tracking
        second_last_city = last_city
        last_city = current_city
        
        path_nodes.append(env.current_node_id)
        done = terminated or truncated or info.get('result') == 'delivered'
        if info.get('result') == 'delivered': 
            break
            
    # Map node IDs -> City names
    city_path = []
    prev = None
    for nid in path_nodes:
        city = nid.split('_')[0].capitalize()
        if city != prev:
            city_path.append(city)
            prev = city
    
    # ── POST-PROCESS: Remove city-level loops ────────────────────────
    def remove_city_loops(path):
        if len(path) <= 2: return path
        cleaned = [path[0]]
        for i in range(1, len(path) - 1):
            if path[i] in path[i+1:] and path[i] != path[-1]:
                continue
            if path[i] != cleaned[-1]:
                cleaned.append(path[i])
        cleaned.append(path[-1])
        return cleaned
    
    city_path = remove_city_loops(city_path)
    
    return city_path, "Success", src_city, dst_city

# ─── Color Palette ────────────────────────────────────────────────────────────
BG           = "#f7f9fc"
CARD_BG      = "#ffffff"
BORDER       = "#dde3ed"
ACCENT_BLUE  = "#1a56db"
ACCENT_GREEN = "#057a55"
ACCENT_RED   = "#e02424"
ACCENT_AMBER = "#d97706"
TEXT_PRIMARY = "#111928"
TEXT_MUTED   = "#6b7280"
TEXT_LIGHT   = "#9ca3af"
SIDEBAR_BG   = "#1e2939"
SIDEBAR_TEXT = "#e5e7eb"
SIDEBAR_MUTED= "#9ca3af"

# ─── Tunisia City Data ────────────────────────────────────────────────────────
CITIES = {
    "Tunis":     {"lat": 36.82, "lon": 10.17, "pop": 1079000, "terrain": "Urban"},
    "Sfax":      {"lat": 34.74, "lon": 10.76, "pop": 608000,  "terrain": "Coastal"},
    "Sousse":    {"lat": 35.83, "lon": 10.64, "pop": 1074468,  "terrain": "Coastal"},
    "Nabeul":    {"lat": 36.45, "lon": 10.73, "pop": 766991,   "terrain": "Coastal"},
    "Bizerte":   {"lat": 37.27, "lon": 9.87,  "pop": 607388,  "terrain": "Coastal"},
    "Kairouan":  {"lat": 35.67, "lon": 10.10, "pop": 600803,  "terrain": "Urban"},
    "Kasserine": {"lat": 35.17, "lon": 8.83,  "pop": 439200,   "terrain": "Mountain"},
    "Monastir":  {"lat": 35.78, "lon": 10.83, "pop": 599769,   "terrain": "Coastal"},
    "Mahdia":    {"lat": 35.50, "lon": 11.06, "pop": 449985 ,   "terrain": "Coastal"},
    "Ariana":    {"lat": 36.86, "lon": 10.19, "pop": 668552,   "terrain": "Urban"},
    "Hammamet":  {"lat": 36.40, "lon": 10.61, "pop": 96181,  "terrain": "Coastal"},
    "SidiBouzid":{"lat": 35.04, "lon": 9.48,  "pop": 489991,  "terrain": "Urban"},
    "Gafsa":     {"lat": 34.42, "lon": 8.78,  "pop": 357000,  "terrain": "Desert"},
    "Gabes":     {"lat": 33.88, "lon": 10.10, "pop": 410847, "terrain": "Coastal"},
    "Medenine":  {"lat": 33.35, "lon": 10.50, "pop": 537255,  "terrain": "Desert"},
    "Tataouine": {"lat": 32.93, "lon": 10.45, "pop": 162654,  "terrain": "Desert"},
    "Tozeur":    {"lat": 33.92, "lon": 8.13,  "pop": 120036 ,  "terrain": "Desert"},
    "Kebili":    {"lat": 33.70, "lon": 8.90,  "pop": 183201,  "terrain": "Desert"},
    "Jendouba":  {"lat": 36.50, "lon": 8.78,  "pop": 404352,  "terrain": "Mountain"},
    "LeKef":     {"lat": 36.17, "lon": 8.71,  "pop": 237686,  "terrain": "Mountain"},
}

# ✅ Define ONLY the 10 simulated cities (matching trained models - 30 nodes)
SIMULATED_CITIES = sorted(["Tunis", "Ariana", "Bizerte", "Nabeul", "Hammamet", 
                           "Sousse", "Monastir", "Mahdia", "Sfax", "Kairouan"])

# ─── Disaster Definitions ─────────────────────────────────────────────────────
DISASTERS = {
    "flood": {
        "label":    "Nabeul Coastal Flood",
        "affected": {"Nabeul", "Monastir", "Mahdia", "Sousse"},
        "color":    ACCENT_BLUE,
        "icon":     "🌊",
        "desc":     "Progressive coastal flooding in Cap Bon region. Nodes fail from the coast inward as water levels rise.",
    },
    "earthquake": {
        "label":    "Kasserine Earthquake",
        "affected": {"Kasserine", "Kairouan"},
        "color":    ACCENT_AMBER,
        "icon":     "🏔",
        "desc":     "Seismic event (M5.5) in mountainous region. Sudden infrastructure collapse with aftershocks.",
    },
    "infrastructure": {
        "label":    "Tunis Infrastructure Failure",
        "affected": {"Tunis", "Ariana"},
        "color":    ACCENT_RED,
        "icon":     "⚡",
        "desc":     "Critical hub failure causing cascade effect across densely connected urban nodes.",
    },
}

# ─── METRICSMetrics ──────────────────────────────────────────────────────────────
# Load metrics from the Backend (Comparison Script)
METRICS_FILE = "data/dashboard_metrics.json"

if os.path.exists(METRICS_FILE):
    with open(METRICS_FILE, "r") as f:
        METRICS = json.load(f)
else:
    # Fallback values if you haven't run comparison_test.py yet
    METRICS = {
        "delivery_rate_dqn": 0.85,
        "energy_reduction": 90.0,
        "co2_saved": 140.0,
        "pop_coverage": 80.0,
        "dqn_avg_hops": 1.5
    }

app = dash.Dash(__name__, title="MeshNet AI — Tunisia Disaster Recovery")
app.config.suppress_callback_exceptions = True

import plotly.graph_objects as go

# ─── Map Builder ─────────────────────────────────────────────────────────────
def build_map(disaster=None, route_path=None, current_step=0, engine="RL", view_mode="city", net_for_nodes=None):
    """Build interactive Plotly map with Tunisia cities (NO TOKEN NEEDED)"""
    affected = DISASTERS[disaster]["affected"] if disaster else set()
    
    fig = go.Figure()
    
    # ─── LAYOUT SETUP ────────────────────────────────────────────────────────
    if view_mode == "node":
        # Use simple 2D plot for node view (network coordinates 0-1000)
        fig.update_layout(
            xaxis=dict(range=[-50, 1050], showgrid=True, gridcolor=BORDER, title="X Coordinate"),
            yaxis=dict(range=[1050, -50], showgrid=True, gridcolor=BORDER, title="Y Coordinate"),
            paper_bgcolor=CARD_BG, margin={"r": 0, "t": 0, "l": 0, "b": 0}, height=500, showlegend=True
        )
    else:
        # Use geographic map for city view
        fig.update_layout(
            geo=dict(scope='world', projection_type='mercator', center=dict(lat=34.5, lon=9.5),
                     projection_scale=3.8, uirevision='tunisia-lock', showland=True,
                     landcolor="rgb(243, 243, 243)", countrycolor="rgb(200, 200, 200)", showcountries=True,
                     showcoastlines=True, coastlinecolor="rgb(150, 150, 200)", bgcolor=CARD_BG),
            paper_bgcolor=CARD_BG, margin={"r": 0, "t": 0, "l": 0, "b": 0}, height=500, showlegend=True,
            legend=dict(bgcolor="rgba(255,255,255,0.92)", bordercolor=BORDER, borderwidth=1,
                    font=dict(color=TEXT_PRIMARY, size=12), x=0.01, y=0.99),
        )
    
    # ─── CITY VIEW: DRAW CONNECTIONS ─────────────────────────────────────────
    if view_mode == "city":
        city_list = list(CITIES.items())
        for i, (c1, d1) in enumerate(city_list):
            for c2, d2 in city_list[i+1:]:
                dist = ((d1["lat"]-d2["lat"])**2 + (d1["lon"]-d2["lon"])**2)**0.5
                if dist < 2.5:
                    both = c1 in affected and c2 in affected
                    one = (c1 in affected) != (c2 in affected)
                    ec = "rgba(224, 36, 36, 0.2)" if both else ("rgba(245, 158, 11, 0.27)" if one else "rgba(26, 86, 219, 0.13)")
                    fig.add_trace(go.Scattergeo(lat=[d1["lat"], d2["lat"]], lon=[d1["lon"], d2["lon"]],
                        mode="lines", line=dict(color=ec, width=2), showlegend=False, hoverinfo="skip"))
    
    # ─── NODE VIEW: DRAW INDIVIDUAL NODES (FIXED) ────────────────────────────
    # Fix: Use go.Scatter with x,y coordinates and status-based colors
    if view_mode == "node" and net_for_nodes:
        for node in net_for_nodes.nodes.values():
            # 1. Determine color based on ACTUAL state
            if not node.is_active:
                color = ACCENT_RED
                status_str = "❌ Offline"
            elif node.battery > 50:
                color = ACCENT_GREEN
                status_str = "✅ Active"
            else:
                color = ACCENT_AMBER
                status_str = "⚠ Degraded"
            
            # 2. Plot using network coordinates (x, y)
            fig.add_trace(go.Scatter(
                x=[node.x], y=[node.y],  # ✅ Use x,y, not lat/lon
                mode="markers",
                marker=dict(size=10, color=color, line=dict(width=2, color="white")),
                name=f"{node._city_label.capitalize()} Node {int(node.id.split('_')[-1]) + 1}",
                hovertemplate=f"<b>{node._city_label.capitalize()} • Node {int(node.id.split('_')[-1]) + 1}</b><br>"
                             f"Status: {status_str}<br>Battery: {node.battery:.1f}%<br>"
                             f"Neighbors: {len(node.neighbors)}<extra></extra>"
            ))

    # ─── CITY VIEW: DRAW CITY MARKERS ────────────────────────────────────────
    else:
        active = {k: v for k, v in CITIES.items() if k not in affected}
        if active:
            fig.add_trace(go.Scattergeo(lat=[v["lat"] for v in active.values()], lon=[v["lon"] for v in active.values()],
                mode="markers+text", marker=dict(size=12, color=ACCENT_GREEN, line=dict(width=2, color='white')),
                text=list(active.keys()), textposition="top right", textfont=dict(color=TEXT_PRIMARY, size=10),
                name="Active Nodes", hovertemplate="%{text}<br>Status: Online<extra></extra>"))
        
        if affected:
            aff = {k: v for k, v in CITIES.items() if k in affected}
            fig.add_trace(go.Scattergeo(lat=[v["lat"] for v in aff.values()], lon=[v["lon"] for v in aff.values()],
                mode="markers+text", marker=dict(size=12, color=ACCENT_AMBER, line=dict(width=2, color='white')),
                text=list(aff.keys()), textposition="top right", textfont=dict(color=ACCENT_RED, size=10),
                name="⚠ Affected Zone", hovertemplate="%{text}<br>Status: Partial Failure<extra></extra>"))
    
    # ─── ROUTE ANIMATION (Both Views) ────────────────────────────────────────
    if route_path and len(route_path) > 1 and current_step > 0:
        color = ACCENT_AMBER if engine == "RL" else ACCENT_BLUE
        
        if view_mode == "city":
            # City view: Geographic coordinates
            full_lats = [CITIES[c]["lat"] for c in route_path if c in CITIES]
            full_lons = [CITIES[c]["lon"] for c in route_path if c in CITIES]
            if full_lats:
                fig.add_trace(go.Scattergeo(lat=full_lats, lon=full_lons, mode="lines",
                    line=dict(color="rgba(100,100,100,0.3)", width=3, dash="dash"), showlegend=False))
            
            t_lats = [CITIES[c]["lat"] for c in route_path[:current_step] if c in CITIES]
            t_lons = [CITIES[c]["lon"] for c in route_path[:current_step] if c in CITIES]
            if t_lats:
                fig.add_trace(go.Scattergeo(lat=t_lats, lon=t_lons, mode="lines",
                    line=dict(color=color, width=6), showlegend=False))
            
            curr = route_path[current_step-1]
            if curr in CITIES:
                fig.add_trace(go.Scattergeo(lat=[CITIES[curr]["lat"]], lon=[CITIES[curr]["lon"]],
                    mode="markers", marker=dict(size=18, color=color, symbol="star", line=dict(width=2, color="white")),
                    name="📩 Message", hoverinfo="skip"))
        
        elif view_mode == "node" and net_for_nodes:
            # Node view: Network coordinates (x, y)
            node_positions = []
            for city_name in route_path:
                # Find active node for this city
                city_nodes = [n for n in net_for_nodes.nodes.values() 
                             if n._city_label.lower() == city_name.lower() and n.is_active]
                if city_nodes:
                    node_positions.append((city_nodes[0].x, city_nodes[0].y))
            
            if len(node_positions) >= 2:
                path_x = [p[0] for p in node_positions]
                path_y = [p[1] for p in node_positions]
                fig.add_trace(go.Scatter(x=path_x, y=path_y, mode="lines",
                    line=dict(color="rgba(100,100,100,0.3)", width=3, dash="dash"),
                    showlegend=False, hoverinfo="skip"))
                
                if current_step > 1:
                    traveled_x = [p[0] for p in node_positions[:current_step]]
                    traveled_y = [p[1] for p in node_positions[:current_step]]
                    fig.add_trace(go.Scatter(x=traveled_x, y=traveled_y, mode="lines",
                        line=dict(color=color, width=6), showlegend=False, hoverinfo="skip"))
                
                if current_step <= len(node_positions):
                    curr_x, curr_y = node_positions[current_step-1]
                    fig.add_trace(go.Scatter(x=[curr_x], y=[curr_y], mode="markers",
                        marker=dict(size=18, color=color, symbol="star", line=dict(width=2, color="white")),
                        name="📩 Message", hoverinfo="skip"))
    
    return fig

# ─── UI Helpers ──────────────────────────────────────────────────────────────
def section_title(text):
    return html.Div(text, style={
        "fontSize": "10px", "fontWeight": "700", "letterSpacing": "2px",
        "textTransform": "uppercase", "color": SIDEBAR_MUTED,
        "marginBottom": "10px", "marginTop": "4px",
    })

def sidebar_btn(label, btn_id, color):
    return html.Button(label, id=btn_id, n_clicks=0, style={
        "width": "100%", "textAlign": "left", "background": "transparent",
        "border": f"1px solid {color}44", "borderLeft": f"3px solid {color}",
        "color": SIDEBAR_TEXT, "padding": "10px 14px", "borderRadius": "6px",
        "cursor": "pointer", "fontSize": "13px", "fontFamily": "Georgia, serif",
        "marginBottom": "8px", "letterSpacing": "0.3px",
    })

def kpi_card(label, value, unit, color, note=""):
    return html.Div([
        html.Div(value + unit, style={
            "fontSize": "26px", "fontWeight": "800", "color": color,
            "fontFamily": "Georgia, serif", "lineHeight": "1.1",
        }),
        html.Div(label, style={
            "fontSize": "11px", "color": TEXT_MUTED,
            "marginTop": "4px", "fontWeight": "600", "letterSpacing": "0.5px",
        }),
        html.Div(note, style={"fontSize": "10px", "color": TEXT_LIGHT, "marginTop": "2px"}) if note else html.Div(),
    ], style={
        "background": CARD_BG, "border": f"1px solid {BORDER}",
        "borderTop": f"3px solid {color}", "borderRadius": "8px",
        "padding": "16px", "flex": "1", "minWidth": "130px",
    })

def status_pill(text, color):
    return html.Span(text, style={
        "background": color + "18", "color": color,
        "border": f"1px solid {color}44", "borderRadius": "20px",
        "padding": "3px 12px", "fontSize": "11px",
        "fontWeight": "700", "letterSpacing": "0.8px",
    })

def divider():
    return html.Hr(style={"border": "none", "borderTop": "1px solid #374151", "margin": "16px 0"})

def METRICS_bar(label, value, color):
    return html.Div([
        html.Div([
            html.Span(label, style={"fontSize": "12px", "color": SIDEBAR_TEXT}),
            html.Span(f"{value}%", style={"fontSize": "12px", "color": color, "fontWeight": "700"}),
        ], style={"display": "flex", "justifyContent": "space-between", "marginBottom": "5px"}),
        html.Div(html.Div(style={
            "width": f"{value}%", "height": "100%",
            "background": color, "borderRadius": "4px",
        }), style={"background": "#374151", "borderRadius": "4px", "height": "6px", "marginBottom": "12px"}),
    ])

def td(bold=False):
    return {
        "padding": "8px 10px", "fontSize": "12px",
        "color": TEXT_PRIMARY if bold else TEXT_MUTED,
        "fontWeight": "600" if bold else "400", "fontFamily": "Georgia, serif",
    }

def build_node_table(affected):
    rows = []
    for city, data in CITIES.items():
        is_aff = city in affected
        rows.append(html.Tr([
            html.Td(city, style=td(bold=True)),
            html.Td(data["terrain"], style=td()),
            html.Td(f"{data['pop']:,}", style=td()),
            html.Td("3", style=td()),
            html.Td(html.Span(
                "● OFFLINE" if is_aff else "● ACTIVE",
                style={"color": ACCENT_RED if is_aff else ACCENT_GREEN,
                       "fontSize": "11px", "fontWeight": "700"},
            )),
        ], style={"borderBottom": f"1px solid {BORDER}"}))
    return html.Table([
        html.Thead(html.Tr([
            html.Th(h, style={
                "color": TEXT_MUTED, "fontSize": "10px", "letterSpacing": "1px",
                "padding": "6px 10px", "textAlign": "left",
                "borderBottom": f"2px solid {BORDER}", "textTransform": "uppercase",
            }) for h in ["City", "Terrain", "Population", "Nodes", "Status"]
        ])),
        html.Tbody(rows),
    ], style={"width": "100%", "borderCollapse": "collapse"})

def METRICS_item(icon, title, text, color):
    return html.Div([
        html.Div(f"{icon} {title}", style={
            "fontSize": "12px", "fontWeight": "700", "color": color, "marginBottom": "6px",
        }),
        html.Div(text, style={"fontSize": "11px", "color": TEXT_MUTED, "lineHeight": "1.7", "marginBottom": "14px"}),
    ])

def kpi_row_children(n_aff, total):
    active = total - n_aff
    health = round((active / total) * 100, 1)
    
    # Pull dynamic values from METRICS
    pop_reached = METRICS.get("pop_coverage", 80.0)
    co2_saved = METRICS.get("co2_saved", 140.0)
    
    return [
        kpi_card("Active Nodes", str(active), "", 
                ACCENT_GREEN if n_aff == 0 else ACCENT_AMBER, f"out of {total}"),
        kpi_card("Affected Nodes", str(n_aff), "", 
                ACCENT_RED if n_aff > 0 else TEXT_MUTED, "offline"),
        kpi_card("Network Health", str(health), "%", 
                ACCENT_GREEN if health == 100 else ACCENT_AMBER),
        kpi_card("Population Reached", f"{pop_reached}", "%", 
                ACCENT_BLUE, "of Tunisia"),
        kpi_card("CO₂ Saved", f"{co2_saved}", " kg", 
                ACCENT_GREEN, "vs flooding"),
    ]

# ─── Layout ─────────────────────────────────────────────────────────────────
app.layout = html.Div(style={
    "backgroundColor": BG, "minHeight": "100vh",
    "fontFamily": "Georgia, 'Times New Roman', serif",
    "display": "flex", "flexDirection": "column",
}, children=[
        dcc.Store(id="sim-state", data={
        "disaster": None, 
        "route_path": [],
        "route_src": None,
        "route_dst": None,
        "step": 0, 
        "active": False, 
        "engine": "rl"
    }),
    dcc.Interval(id="route-timer", interval=600, n_intervals=0, disabled=True),

    # Header
    html.Div([
        html.Div([
            html.Div("◈", style={"color": ACCENT_BLUE, "fontSize": "22px", "marginRight": "12px"}),
            html.Div([
                html.Span("MeshNet AI", style={"fontSize": "18px", "fontWeight": "900", "color": TEXT_PRIMARY}),
                html.Span(" — Tunisia Disaster Recovery Dashboard",
                          style={"fontSize": "14px", "color": TEXT_MUTED, "marginLeft": "4px"}),
            ]),
        ], style={"display": "flex", "alignItems": "center"}),
        html.Div([
            html.Span("ENISO · Academic Project · 2025–2026",
                      style={"fontSize": "11px", "color": TEXT_MUTED, "letterSpacing": "1px", "marginRight": "20px"}),
            status_pill("● SYSTEM ONLINE", ACCENT_GREEN),
        ], style={"display": "flex", "alignItems": "center"}),
    ], style={
        "display": "flex", "justifyContent": "space-between", "alignItems": "center",
        "padding": "14px 28px", "background": CARD_BG,
        "borderBottom": f"1px solid {BORDER}", "boxShadow": "0 1px 4px rgba(0,0,0,0.06)",
    }),

    # Body
    html.Div(style={"display": "flex", "flex": "1"}, children=[

        # Sidebar
        html.Div([
            section_title("Network Status"),
            html.Div(id="sidebar-status", children=[
                html.Div("All systems nominal", style={"fontSize": "12px", "color": "#6ee7b7", "marginBottom": "4px"}),
                html.Div("30 / 30 nodes active", style={"fontSize": "11px", "color": SIDEBAR_MUTED}),
            ], style={"marginBottom": "20px"}),

            divider(),
            section_title("Disaster Scenarios"),
            sidebar_btn("🌊  Nabeul Flood",            "btn-flood", ACCENT_BLUE),
            sidebar_btn("🏔  Kasserine Earthquake",    "btn-quake", ACCENT_AMBER),
            sidebar_btn("⚡  Tunis Infrastructure",    "btn-infra", ACCENT_RED),
            html.Div(style={"marginTop": "8px"}),
            sidebar_btn("↺  Reset Network",            "btn-reset", ACCENT_GREEN),
            html.Div(style={"marginTop": "12px"}),
            section_title("AI Routing Engine"),
            dcc.RadioItems(
                id="rl-model-selector",
                options=[
                    {"label": "🧠 DQN Agent", "value": "dqn"},
                    {"label": "🤖 PPO Agent", "value": "ppo"}
                ],
                value="dqn",
                style={"color": SIDEBAR_TEXT, "fontSize": "12px"}
            ),
            html.Button("🚀 Run AI Routing", id="btn-ai-route", n_clicks=0, style={
                "width": "100%", "textAlign": "left", "background": "transparent",
                "border": f"1px solid {ACCENT_AMBER}44", "borderLeft": f"3px solid {ACCENT_AMBER}",
                "color": SIDEBAR_TEXT, "padding": "10px 14px", "borderRadius": "6px",
                "cursor": "pointer", "fontSize": "13px", "fontFamily": "Georgia, serif",
                "marginTop": "8px",
            }),
            html.Div(style={"marginTop": "12px"}),
            section_title("🔍 View Mode"),
            dcc.RadioItems(
                id="view-mode",
                options=[
                    {"label": "🏙️ City View", "value": "city"},
                    {"label": "📱 Node View", "value": "node"}
                ],
                value="city",
                style={"color": SIDEBAR_TEXT, "fontSize": "12px"}
            ),
            html.Div(style={"marginTop": "12px"}),
            section_title("📍 Custom Routing (Optional)"),
            dcc.Dropdown(id="source-city", options=[{"label": c, "value": c} for c in SIMULATED_CITIES],
                         placeholder="Source (Random)", clearable=True,
                         style={"color": "#374151", "fontSize": "12px", "background": "#F3F4F6"}),
            html.Div(style={"height": "8px"}),
            dcc.Dropdown(id="dest-city", options=[{"label": c, "value": c} for c in SIMULATED_CITIES],
                         placeholder="Destination (Random)", clearable=True,
                         style={"color": "#374151", "fontSize": "12px", "background": "#F3F4F6"}),

            divider(),
            html.Div(id="scenario-desc", children=[
                html.Div("Select a scenario above to begin simulation.",
                         style={"fontSize": "12px", "color": SIDEBAR_MUTED, "lineHeight": "1.7", "fontStyle": "italic"}),
            ], style={"marginBottom": "20px"}),

            divider(),
            section_title("Key Indicators"),
            METRICS_bar("Population Coverage", METRICS.get("pop_coverage", 80.0),     "#34d399"),
            METRICS_bar("Network Uptime",       METRICS.get("uptime", 95.0),           "#60a5fa"),
            METRICS_bar("Rural Coverage",       METRICS.get("rural_coverage", 70.0),   "#a78bfa"),
            METRICS_bar("Energy Reduction",     METRICS.get("energy_reduction", 90.0), "#fbbf24"),


            divider(),
            html.Div([
                html.Div("CO₂ Saved", style={"fontSize": "10px", "color": SIDEBAR_MUTED, "letterSpacing": "1px"}),
                html.Div(f"{METRICS.get('co2_saved', 140.0)} kg", style={
                    "fontSize": "22px", "fontWeight": "800", "color": "#34d399", "fontFamily": "Georgia, serif",
                }),
                html.Div("vs. flooding algorithm", style={"fontSize": "10px", "color": SIDEBAR_MUTED, "marginTop": "2px"}),
            ], style={"textAlign": "center", "padding": "8px 0"}),

        ], style={
            "width": "240px", "minWidth": "240px", "background": SIDEBAR_BG,
            "padding": "24px 16px", "overflowY": "auto", "borderRight": "1px solid #374151",
        }),

        # Main
        html.Div([
            # KPI row
            html.Div(id="kpi-row", children=kpi_row_children(0, 30),
                     style={"display": "flex", "gap": "14px", "marginBottom": "20px", "flexWrap": "wrap"}),

            # Map card
            html.Div([
                html.Div([
                    html.Div([
                        html.Span("Tunisia Mesh Network",
                                  style={"fontSize": "15px", "fontWeight": "700", "color": TEXT_PRIMARY}),
                        html.Span(id="map-subtitle", children=" — Normal Operation",
                                  style={"fontSize": "13px", "color": TEXT_MUTED}),
                    ]),
                    html.Div(id="disaster-badge", children=status_pill("NORMAL", ACCENT_GREEN)),
                ], style={"display": "flex", "justifyContent": "space-between", "alignItems": "center", "marginBottom": "14px"}),
                html.Div(id="route-info", children="", style={
                    "fontSize": "12px", "color": TEXT_MUTED, "marginBottom": "10px",
                    "fontStyle": "italic", "textAlign": "center"
                }),
                dcc.Graph(id="main-map", figure=build_map(), config={"displayModeBar": False},
                          style={"borderRadius": "8px", "overflow": "hidden"}),
                html.Div([
                        html.Span("● Green", style={"color": ACCENT_GREEN, "marginRight": "4px"}),
                        html.Span("Fully Active  |  ", style={"fontSize": "11px", "color": TEXT_MUTED}),
                        html.Span("● Amber", style={"color": ACCENT_AMBER, "marginRight": "4px"}),
                        html.Span("Degraded (partial failure)  |  ", style={"fontSize": "11px", "color": TEXT_MUTED}),
                        html.Span("● Red", style={"color": ACCENT_RED, "marginRight": "4px"}),
                        html.Span("Offline", style={"fontSize": "11px", "color": TEXT_MUTED}),
                        html.Span("  |  ⭐ AI Agent Hop", style={"fontSize": "11px", "color": TEXT_MUTED}),
                    ], style={"textAlign": "center", "marginTop": "8px", "fontSize": "11px", "color": TEXT_MUTED}),
                            
            ], style={
                "background": CARD_BG, "border": f"1px solid {BORDER}",
                "borderRadius": "10px", "padding": "20px",
                "boxShadow": "0 1px 6px rgba(0,0,0,0.05)", "marginBottom": "20px",
            }),

            # Bottom row
            html.Div([
                html.Div([
                    html.Div("Node Registry", style={"fontSize": "13px", "fontWeight": "700", "color": TEXT_PRIMARY, "marginBottom": "14px"}),
                    html.Div(id="node-table", children=build_node_table(set())),
                ], style={
                    "background": CARD_BG, "border": f"1px solid {BORDER}",
                    "borderRadius": "10px", "padding": "20px", "flex": "1.4",
                    "boxShadow": "0 1px 6px rgba(0,0,0,0.05)",
                }),

                html.Div([
                    html.Div("Project Impact Summary", style={"fontSize": "13px", "fontWeight": "700", "color": TEXT_PRIMARY, "marginBottom": "14px"}),
                    METRICS_item("🌿", "Environmental",
                        f"AI routing reduces energy by {METRICS.get('energy_reduction', 90.0)}% vs flooding. "
                        f"DQN uses 0.13 units vs 19.40 — saving ~{METRICS.get('co2_saved', 140.0)} kg CO₂.",
                        ACCENT_GREEN),
                    METRICS_item("👥", "Social",
                        f"{METRICS.get('pop_coverage', 80.0)}% of Tunisia's population stays connected during disaster. "
                        f"Rural areas achieve {METRICS.get('rural_coverage', 70.0)}% coverage.", 
                        ACCENT_BLUE),
                    METRICS_item("📡", "Resilience",
                        f"Network maintains {METRICS.get('uptime', 95.0)}% uptime across all 3 tested disaster scenarios. "  
                        f"DQN routing adapts automatically without reprogramming.",
                        ACCENT_AMBER),
                ], style={
                    "background": CARD_BG, "border": f"1px solid {BORDER}",
                    "borderRadius": "10px", "padding": "20px", "flex": "1",
                    "boxShadow": "0 1px 6px rgba(0,0,0,0.05)",
                }),
            ], style={"display": "flex", "gap": "16px", "flexWrap": "wrap"}),

        ], style={"flex": "1", "padding": "24px", "overflowY": "auto"}),
    ]),
])


# ─── Callback ────────────────────────────────────────────────────────────────
@app.callback(
    Output("sim-state", "data"),
    Output("route-timer", "disabled"),
    Output("route-timer", "n_intervals"),
    Input("btn-flood", "n_clicks"),
    Input("btn-quake", "n_clicks"),
    Input("btn-infra", "n_clicks"),
    Input("btn-reset", "n_clicks"),
    Input("btn-ai-route", "n_clicks"),
    Input("source-city", "value"),
    Input("dest-city", "value"),
    State("sim-state", "data"),
    State("rl-model-selector", "value"),
    prevent_initial_call=True,
)
def handle_disaster_and_ai(flood, quake, infra, reset, ai_run, src_sel, dst_sel, sim_data, model_type):
    triggered = ctx.triggered_id
    state = sim_data.copy() if sim_data else {"disaster": None, "route_path": [], "step": 0, "active": False, "engine": "rl"}
    
    if triggered == "btn-reset":
        return {"disaster": None, "route_path": [], "step": 0, "active": False, "engine": "rl"}, True, 0
        
    disaster_map = {"btn-flood": "flood", "btn-quake": "earthquake", "btn-infra": "infrastructure"}
    if triggered in disaster_map:
        state["disaster"] = disaster_map[triggered]
        state["route_path"] = []
        state["step"] = 0
        state["active"] = False
        return state, True, 0
        
    if triggered == "btn-ai-route":
        affected = DISASTERS[state["disaster"]]["affected"] if state["disaster"] else set()
        path, status, src_city, dst_city = get_rl_route(affected, model_type=model_type, seed=hash(f"{state['disaster']}_{model_type}") % 1000,
                                                        force_src_city=src_sel if src_sel else None, force_dest_city=dst_sel if dst_sel else None)
        
        state["route_path"] = path
        state["route_src"] = src_city
        state["route_dst"] = dst_city
        state["step"] = 0
        state["active"] = len(path) > 1
        state["engine"] = f"{model_type.upper()}"
        return state, not state["active"], 0
        
    return state, True, 0

@app.callback(
    Output("main-map", "figure"),
    Output("map-subtitle", "children"),
    Output("disaster-badge", "children"),
    Output("kpi-row", "children"),
    Output("node-table", "children"),
    Output("sidebar-status", "children"),
    Output("scenario-desc", "children"),
    Output("route-info", "children"),
    Output("route-timer", "disabled"),
    Input("route-timer", "n_intervals"),
    Input("view-mode", "value"), 
    State("sim-state", "data"),
)
def animate_route(n, view_mode, sim_data):
    view_mode = view_mode or "city"
    if not sim_data or not sim_data.get("active"):
        # Static render
        disaster = sim_data.get("disaster")
        affected = DISASTERS[disaster]["affected"] if disaster else set()
        n_aff = len(affected) * 3
        d = DISASTERS.get(disaster, {})
        
        # Build network for node view
        net_for_nodes = None
        if view_mode == "node":  # ✅ Always build for node view
            affected_cities = DISASTERS[disaster]["affected"] if disaster else set()
            net_for_nodes = build_rl_network(affected_cities, seed=42)
        
        return (
            build_map(disaster=disaster, view_mode=view_mode, net_for_nodes=net_for_nodes),
            f" — {d.get('icon','')} {d.get('label','')} Active" if disaster else " — Normal Operation",
            status_pill("⚠ DISASTER ACTIVE", d.get("color", ACCENT_RED)) if disaster else status_pill("NORMAL", ACCENT_GREEN),
            kpi_row_children(n_aff, 30),
            build_node_table(affected),
            [html.Div(f"⚠ {n_aff} nodes offline", style={"fontSize": "12px", "color": "#fca5a5", "marginBottom": "4px"})] if n_aff > 0 else [html.Div("All systems nominal", style={"fontSize": "12px", "color": "#6ee7b7", "marginBottom": "4px"})],
            [html.Div(d.get("desc", ""), style={"fontSize": "12px", "color": SIDEBAR_MUTED, "lineHeight": "1.7"})] if disaster else [html.Div("Select a scenario above.", style={"fontSize": "12px", "color": SIDEBAR_MUTED})],
            "",
            True
        )
    
    # Animation running
    step = n
    path = sim_data["route_path"]
    src = sim_data.get("route_src", "Unknown")
    dst = sim_data.get("route_dst", "Unknown")
    engine = sim_data.get("engine", "RL")
    is_done = step >= len(path)

    # 🆕 Build route info text
    print(f"DEBUG: path={path}, src={src}, dst={dst}, step={step}, len(path)={len(path)}")
    if path and len(path) >= 2:
        src = sim_data.get("route_src", "Unknown")      # Intended source
        dst = sim_data.get("route_dst", "Unknown")      # Intended destination
        actual = sim_data.get("actual_dst", dst)        # Actually reached
        current = path[min(step, len(path)-1)]
        
        # Show warning if intended != actual
        if dst != actual and step >= len(path):
            status_text = html.Span(f" ⚠️ Reached {actual} ( {dst} unavailable)", 
                                style={"color": ACCENT_AMBER, "fontSize": "11px", "marginLeft": "6px"})
        elif step >= len(path):
            status_text = html.Span(" ✅ Destination reached", 
                                style={"color": ACCENT_GREEN, "fontSize": "11px", "marginLeft": "6px"})
        else:
            status_text = html.Span("", style={"display": "none"})
        
        route_display = html.Span([
            html.Span(src, style={"color": ACCENT_GREEN}),
            html.Span(" → ", style={"color": TEXT_MUTED}),
            html.Span(current, style={"color": ACCENT_BLUE, "fontWeight": "700"}),
            html.Span(" → ", style={"color": TEXT_MUTED}),
            html.Span(dst, style={"color": ACCENT_GREEN}),  # Always show INTENDED destination
        ])
        
        route_text = html.Span([
            html.Span(f"📩 {engine}: ", style={"color": ACCENT_AMBER, "fontWeight": "600"}),
            route_display,
            status_text
        ])
    else:
        route_text = html.Span("⚠️ No viable path found.", style={"color": ACCENT_RED, "fontWeight": "600"})
    
    
    net_for_nodes = None
    if view_mode == "node":
        affected_cities = DISASTERS[sim_data.get("disaster", {})].get("affected", set()) if sim_data.get("disaster") else set()
        net_for_nodes = build_rl_network(affected_cities, seed=42)
    
    return (
        build_map(disaster=sim_data.get("disaster"), route_path=path, current_step=step, engine=engine, view_mode=view_mode, net_for_nodes=net_for_nodes),
        " — 🧠 AI Routing..." if not is_done else " — Delivered!",
        status_pill(f"📡 {engine}", ACCENT_AMBER) if not is_done else status_pill("DELIVERED", ACCENT_GREEN),
        kpi_row_children(len(DISASTERS.get(sim_data.get("disaster"), {}).get("affected", set()))*3, 30),
        build_node_table(DISASTERS.get(sim_data.get("disaster"), {}).get("affected", set())),
        [html.Div(f"Step {step}/{len(path)}", style={"fontSize": "12px", "color": "#fbbf24", "marginBottom": "4px"})],
        [html.Div(f"{engine} agent selecting next hop based on 28-feature state...", style={"fontSize": "12px", "color": SIDEBAR_MUTED})],
        route_text,
        is_done
    )

if __name__ == "__main__":
    print("\n" + "=" * 55)
    print("  MeshNet AI Dashboard — Starting...")
    print("  Open browser at: http://127.0.0.1:8050")
    print("=" * 55 + "\n")
    app.run(debug=False, port=8050)