"""
CASEVAC MVerzeide Monte Carlo simulation.

This script runs CASEVAC simulation scenarios over a Monte Carlo design and writes
scenario-level outputs for later analysis. It shares the same model structure as
the visualization script, but focuses on repeated seeded runs and batch metrics.
"""

#region Libraries
# This block imports all dependencies used for simulation, graph routing, random sampling, and batch output.
print("Start importing Libraries")
import os, json, time, traceback
from itertools import product
import random
import matplotlib.lines as mlines
import math
import agentpy as ap
import osmnx as ox
import networkx as nx
import numpy as np
from matplotlib.colors import ListedColormap
from scipy.ndimage import gaussian_filter
from scipy.spatial import cKDTree
from scipy.stats import skewnorm
from collections import Counter
import heapq


print("Imported all Libraries")
#endregion

#region seed
# This block sets the base random seed so repeated runs can be reproduced.
seed = 52
random.seed(seed)
np.random.seed(seed)
#endregion

#region Loading Map
# This block loads the road network, projects it, assigns travel times, and adds reverse edges for two-way routing.
GRAPH_FILE = "gulpen_polygon_simplified.graphml"
if not os.path.exists(GRAPH_FILE):
    raise FileNotFoundError(GRAPH_FILE)

# 1) Load and project the graph while keeping it as a MultiDiGraph so OSMnx plotting still works.
G = ox.load_graphml(GRAPH_FILE)
G = ox.project_graph(G)

# 2) Ensure every edge has speed_kph and travel_time attributes.
for u, v, k, d in G.edges(keys=True, data=True):
    length = float(d.get("length", 100.0))
    speed = float(d.get("speed_kph", 50.0))
    speed = min(max(speed, 10.0), 130.0)  # clamp
    d["speed_kph"] = speed
    d["travel_time"] = (length / 1000.0) / speed * 60.0  # minutes

# 3) Make the network bidirectional to prevent one-way routing failures.
#    Add reverse edges when they are missing.
edges_to_add = []
for u, v, k, d in G.edges(keys=True, data=True):
    # If no edge exists from v to u, add one.
    if not G.has_edge(v, u):
        dcopy = dict(d)  # Copy attributes, including travel_time, speed_kph, and geometry when available.
        edges_to_add.append((v, u, dcopy))

for u2, v2, dcopy in edges_to_add:
    G.add_edge(u2, v2, **dcopy)

nodes = list(G.nodes)
print("Map is imported (MultiDiGraph) + travel_time computed + reverse edges added")
#endregion

#region Global Parameters
# This block defines global model settings, grid resolution, casualty parameters, hotspot parameters, and visualization colors.
DISPATCH_HEURISTIC = 6  # 1: severity-nearest/ambulance-first; 2: nearest/ambulance-first; 3: severity-nearest/helicopter-first; 4: nearest/helicopter-first; 5: ambulance-shortest and helicopter-severity-nearest/ambulance-first; 6: same target rules as 5/helicopter-first.
USE_CLAIRVOYANT_COORDINATOR = False  # True = clairvoyant, False = myopic.
Base_Policy = "SEVERITY"  # "SEVERITY" or "FIFO".
PLATFORM_DOCTRINE = "HARD"  # "SOFT" or "HARD".


xs = [G.nodes[n]["x"] for n in G.nodes]
ys = [G.nodes[n]["y"] for n in G.nodes]
x_min, x_max = min(xs), max(xs)
y_min, y_max = min(ys), max(ys)

Dx, Dy = 50, 50  # resolution in meters
x = np.arange(x_min, x_max, Dx)
y = np.arange(y_min, y_max, Dy)
X, Y = np.meshgrid(x, y)
points = np.column_stack([X.ravel(), Y.ravel()])
N_seeds = 175
NO_GO_TILE_CLASS = 2 # Red tiles are no-go areas.
frontline_angle = np.random.uniform(0,360) # Frontline angle in degrees (0 = attack from the south, 180 = attack from the north).
TIME_STEPS = 300

# Fight intensity & flipping parameters
F_high = 0.75  
F_max = 4.0
B0 = 2250.0   # Frontline width


# Casualty Generation
xi_casualty = 0.005
Lambda_c = 0.4

# Triage Probabilities
TRIAGE_LABELS = ["green", "yellow", "red", "black"]
TRIAGE_PROBS = np.array([0.5, 0.25, 0.25, 0])
TRIAGE_COLORS = {"green": "#2ca02c", "yellow": "#ffcc00", "red": "#d62728", "black": "#000000"}

STAB_PARAMS = {
    "green":  (2.0, 2.5),
    "yellow": (2.0, 7.5),
    "red":    (2.0, 15.0),
    "black":  None
}
DELAY_PARAMS = {
    "green":  (60, 240, 1440),
    "yellow": (30, 90, 180),
    "red":    (10, 30, 60),
    "black":  (0, 0, 0)
}
D_PROB_DETERIORATE = 1

# Hotspot initial sampling ranges 
SIGMA_MIN, SIGMA_MAX = 100.0, 1000.0   
AMP_MIN, AMP_MAX = 0.8, 1.8           
LAMBDA_0 = 25                        

# Dynamic spawn parameters
BASE_SPAWN_PROB = 0.0005  
A_TILE_ON_FLIP_MIN, A_TILE_ON_FLIP_MAX = 1.5, 3 

# Hotspot disappearance when global intensity low
F_low = 0.75 
P0_LOW = 0.05  
BETA_LOW = 0.1   
JITTER = 0       

cmap_tiles = ListedColormap(["#37dc7c", "#FAEB655A", "#f22929"])  
#endregion

#region Adversarial and Map Dynamics Functions
# This block defines grid conversion, frontline dynamics, hotspot generation, casualty generation, and tile-flip metrics.
# -------------------------
# Grid helpers
# -------------------------
def world_to_grid(xw, yw, x_min, y_min, Dx, Dy):
    ix = int((xw - x_min) / Dx)
    iy = int((yw - y_min) / Dy)
    return ix, iy

def grid_to_world(ix, iy, x, y):
    return x[ix], y[iy]

def perlin1d(n, scale=2500, smooth=40):
    """Generate 1D smooth Perlin-like noise for frontline."""
    base = np.random.randn(n)
    smooth_base = gaussian_filter(base, smooth)
    smooth_base -= smooth_base.min()
    smooth_base /= smooth_base.max()
    return scale * (smooth_base - 0.5)

# -------------------------
# Tile Generation
# -------------------------
def generate_tiles(x, y, x_min, y_min, x_max=None, y_max=None, Dx=None, Dy=None, N_seeds=None,
                   frontline_angle_deg=0.0, seed=None):
    np.random.seed(seed)

    # -----------------------------
    # Backwards compatible defaults
    # -----------------------------
    if x_max is None:
        x_max = float(np.max(x))
    if y_max is None:
        y_max = float(np.max(y))

    # Optional safety check for inconsistent x_min/y_min values.
    # Keep supplied values as-is, but fall back to the grid bounds if they are None.
    if x_min is None:
        x_min = float(np.min(x))
    if y_min is None:
        y_min = float(np.min(y))

    # ---------------------------------
    # Frontline orientation
    # ---------------------------------
    theta = np.deg2rad(frontline_angle_deg)

    # Unit vectors
    t_hat = np.array([np.cos(theta), np.sin(theta)])          # along frontline
    n_hat = np.array([-np.sin(theta), np.cos(theta)])         # normal to frontline

    # ---------------------------------
    # Frontline baseline (in normal direction)
    # ---------------------------------
    baseline = 0.5 * (
        np.dot([x_min, y_min], n_hat) +
        np.dot([x_max, y_max], n_hat)
    )

    # Create coordinate along frontline
    X, Y = np.meshgrid(x, y)
    points = np.column_stack([X.ravel(), Y.ravel()])

    s_coord = points @ t_hat
    s_unique = np.linspace(s_coord.min(), s_coord.max(), len(x))

    # Perlin noise along frontline direction
    frontline_offset = perlin1d(len(s_unique), scale=2500, smooth=40)

    # ---------------------------------
    # Random seed points
    # ---------------------------------
    seeds_x = np.random.uniform(x_min, x_max, N_seeds)
    seeds_y = np.random.uniform(y_min, y_max, N_seeds)
    seeds = np.column_stack([seeds_x, seeds_y])

    # ---------------------------------
    # Initial tile classes
    # 0 = friendly, 1 = contested, 2 = enemy
    # ---------------------------------
    B = float(B0)
    p_orange = 0
    tile_class = np.zeros(N_seeds, dtype=int)

    for i in range(N_seeds):
        p = np.array([seeds_x[i], seeds_y[i]])

        # Projection along and normal to frontline
        s = np.dot(p, t_hat)
        n = np.dot(p, n_hat)

        # Find corresponding frontline height
        idx = np.searchsorted(s_unique, s)
        idx = np.clip(idx, 0, len(frontline_offset) - 1)
        frontline_n = baseline + frontline_offset[idx]

        d = n - frontline_n

        if d > B:
            tile_class[i] = 0
        elif d > -B:
            tile_class[i] = 1
        else:
            tile_class[i] = 2

        # random orange reassignment
        if tile_class[i] != 1 and np.random.rand() < p_orange:
            tile_class[i] = 1

    # ---------------------------------
    # Voronoi assignment
    # ---------------------------------
    tree = cKDTree(seeds)
    _, nearest = tree.query(points)
    tile_map = tile_class[nearest].reshape(X.shape)

    # Cells per seed
    cells_per_seed = [np.nonzero(nearest == s)[0] for s in range(N_seeds)]

    frontline_state = {
        "t_hat": t_hat,
        "n_hat": n_hat,
        "baseline": baseline,
        "s_unique": s_unique,
        "frontline_offset": frontline_offset,
        "B": B,
        "delta": 0.0,          # current shift along the normal direction
    }

    return tile_map, tile_class.copy(), seeds, nearest, tree, frontline_state, cells_per_seed


# -------------------------
# Frontline Delta Update
# -------------------------
def update_frontline_delta(frontline_state, F_t, S_t,
                           F_ref=0.75, F_max=4.0,
                           v_max=30.0,       # meters per minute (maximum shift speed)
                           s_scale=1.0,      # scaling factor for tanh applied to S_t
                           delta_max=4000.0  # maximum total shift in meters
                           ):
    """
    Update frontline shift delta along n_hat.

    Desired behavior:
    - S_t < 0  => frontline moves towards friendly side => friendly shrinks
    - S_t > 0  => frontline moves towards enemy side    => no-go (enemy) shrinks
    """

    # intensity factor in [0,1]
    f = (F_t - F_ref) / max(1e-9, (F_max - F_ref))
    f = float(np.clip(f, 0.0, 1.0))

    # direction in [-1,1]  (FLIPPED SIGN vs your original)
    dir_ = float(-np.tanh(S_t / max(1e-9, s_scale)))

    # delta increment
    d_delta = v_max * f * dir_

    frontline_state["delta"] = float(np.clip(
        frontline_state["delta"] + d_delta,
        -delta_max, delta_max
    ))
    return frontline_state["delta"]

def reclassify_seeds_from_frontline(seeds, frontline_state):
    """
    Compute seed tile classes from (baseline + perlin_offset + delta).
    Returns V_dyn (len N_seeds) with classes 0/1/2.
    """
    t_hat = frontline_state["t_hat"]
    n_hat = frontline_state["n_hat"]
    baseline = frontline_state["baseline"]
    s_unique = frontline_state["s_unique"]
    offset = frontline_state["frontline_offset"]
    B = float(frontline_state["B"])
    delta = float(frontline_state["delta"])

    # project seeds to along/normal coordinates
    s = seeds @ t_hat
    n = seeds @ n_hat

    # map each seed to the corresponding perlin offset index
    idx = np.searchsorted(s_unique, s)
    idx = np.clip(idx, 0, len(offset) - 1)

    frontline_n = baseline + offset[idx] + delta
    d = n - frontline_n

    V = np.zeros(len(seeds), dtype=int)
    V[d > B] = 0
    V[(d <= B) & (d >= -B)] = 1
    V[d < -B] = 2
    return V

# -------------------------
# Fight / Adversarial Dynamics
# -------------------------
def simulate_fight_situation(T, mu_F=0.0, rho_F=0.95, beta_F=0.025,
                             shape_alpha_F=3.0, alpha_sigma=0.5, beta_sigma=0.04, gamma_sigma=0.04,
                             F0=0.0, rho_S=0.95, mu_S=0, alpha_S=0.2, beta_S=0.25, S0=0.0, seed=None):
    rs = np.random.RandomState(seed)
    F = np.zeros(T)
    S = np.zeros(T)
    F_prev = float(F0)
    S_prev = float(S0)

    def xi_from_sigma(sigma_t, alpha_F):
        return - sigma_t * (alpha_F / np.sqrt(1.0 + alpha_F**2)) * np.sqrt(2.0 / np.pi)

    for t in range(T):
        sigma_Ft = alpha_sigma + beta_sigma * F_prev + gamma_sigma * abs(S_prev)
        sigma_Ft = max(sigma_Ft, 0.0)
        xi_Ft = xi_from_sigma(sigma_Ft, shape_alpha_F)

        eps_Ft = skewnorm.rvs(a=shape_alpha_F, loc=xi_Ft, scale=sigma_Ft, random_state=rs)
        F_t = rho_F * F_prev + (1.0 - rho_F) * mu_F + beta_F * abs(S_prev) + eps_Ft
        F_t = max(F_t, 0.0)

        var_St = alpha_S + beta_S * F_t
        var_St = max(var_St, 0.0)
        eps_St = rs.normal(loc=0.0, scale=np.sqrt(var_St))
        S_t = rho_S * S_prev + (1.0 - rho_S) * mu_S + eps_St

        F[t] = F_t
        S[t] = S_t
        F_prev, S_prev = F_t, S_t

    return F, S

# -------------------------
# Hotspot Kernels
# -------------------------
_kernel_cache = {}

def make_anisotropic_kernel(sx, sy, theta_rad, Dx, Dy, support_factor=3.0, max_kernel_size=201):
    sx_cells, sy_cells = max(1, sx/Dx), max(1, sy/Dy)
    half = int(np.ceil(support_factor*max(sx_cells, sy_cells)))
    half = min(half, max_kernel_size//2)
    size = 2*half + 1
    ax = np.arange(-half, half+1)
    XX, YY = np.meshgrid(ax, ax)
    c, s = np.cos(-theta_rad), np.sin(-theta_rad)
    Xr = XX*c - YY*s
    Yr = XX*s + YY*c
    kernel = np.exp(-0.5*((Xr/sx_cells)**2 + (Yr/sy_cells)**2))
    kernel /= kernel.max()
    return kernel, half

def get_kernel_cached(sx, sy, theta_rad, Dx, Dy):
    q_sx, q_sy = int(round(sx/16)*16), int(round(sy/16)*16)
    q_theta = int(round(np.degrees(theta_rad)/10)*10)
    key = (q_sx, q_sy, q_theta)
    if key in _kernel_cache:
        return _kernel_cache[key]
    kernel, half = make_anisotropic_kernel(q_sx, q_sy, np.radians(q_theta), Dx, Dy)
    _kernel_cache[key] = (kernel, half)
    return kernel, half

# -------------------------
# Hotspot Sampling & Stamping
# -------------------------
def sample_random_point_in_seed(seed_idx, cells_per_seed, x, y, Dx, Dy, jitter=0.0):
    idxs = cells_per_seed[seed_idx]
    if len(idxs) == 0:
        return None
    linear_idx = np.random.choice(idxs)
    iy, ix = divmod(int(linear_idx), len(x))
    cx = x[ix] + (np.random.rand()-0.5)*Dx*jitter
    cy = y[iy] + (np.random.rand()-0.5)*Dy*jitter
    return cx, cy

def create_hotspot(seed_idx, seeds, cells_per_seed, Dx, Dy, x, y,
                   amp_range=(1.5, 3.0), sigma_range=(SIGMA_MIN, SIGMA_MAX), jitter=0.0):
    loc = sample_random_point_in_seed(seed_idx, cells_per_seed, x, y, Dx, Dy, jitter)
    cx, cy = loc if loc is not None else seeds[seed_idx]
    sx, sy = np.random.uniform(*sigma_range, 2)
    theta = np.random.uniform(0, 2*np.pi)
    amp = np.random.uniform(*amp_range)
    return {'seed_idx': int(seed_idx), 'cx': float(cx), 'cy': float(cy),
            'sx': float(sx), 'sy': float(sy), 'theta': float(theta), 'amp': float(amp)}

def stamp_hotspot_into_grid(grid, hotspot, x_min, y_min, Dx, Dy, tile_mask=None):
    ix, iy = world_to_grid(hotspot['cx'], hotspot['cy'], x_min, y_min, Dx, Dy)
    if ix < 0 or iy < 0 or ix >= grid.shape[1] or iy >= grid.shape[0]:
        return
    if tile_mask is not None and not tile_mask[iy, ix]:
        return
    kernel, half = get_kernel_cached(hotspot['sx'], hotspot['sy'], hotspot['theta'], Dx, Dy)
    ks = kernel.shape[0]
    i0, j0 = iy-half, ix-half
    i1, j1 = i0+ks, j0+ks
    gi0, gj0 = max(0,i0), max(0,j0)
    gi1, gj1 = min(grid.shape[0], i1), min(grid.shape[1], j1)
    ki0, kj0 = gi0-i0, gj0-j0
    ki1, kj1 = ki0+(gi1-gi0), kj0+(gj1-gj0)
    patch = kernel[ki0:ki1, kj0:kj1]
    grid[gi0:gi1, gj0:gj1] += hotspot['amp']*patch

# =========================
# Casualty generation 
# =========================
def sample_triage_label():
    return np.random.choice(TRIAGE_LABELS, p=TRIAGE_PROBS)

def sample_stabilization_time(triage_label):
    if triage_label == "black":
        return 0.0
    k, theta = STAB_PARAMS[triage_label]
    return float(np.random.gamma(shape=k, scale=theta))

def sample_max_delay(triage_label):
    a, mode, b = DELAY_PARAMS[triage_label]
    if a == b:
        return float(a)
    return float(np.random.triangular(a, mode, b))

def casualty_at_t(F_t, S_t, xi = xi_casualty, Lambda_c= Lambda_c):
    Lambda_t = Lambda_c * F_t * np.exp(-xi * S_t)
    Lambda_t = max(Lambda_t,0.0) 
    return np.random.poisson(Lambda_t)

def progress_triage_label(label):
    idx = TRIAGE_LABELS.index(label)
    return TRIAGE_LABELS[min(idx + 1, len(TRIAGE_LABELS)-1)]

# -------------------------
# Tile flip metrics helper
# -------------------------
def compute_flip_counts(V_prev, V_new, classes=(0, 1, 2)):
    """
    Count flips between tile classes for seed-level classes V_prev -> V_new.
    Returns:
      - total_flips: int
      - counts: dict like {"0→1": 12, "1→2": 5, ...}
      - matrix: np.array shape (K,K) where matrix[i,j] = # seeds i->j
    """
    V_prev = np.asarray(V_prev, dtype=int)
    V_new  = np.asarray(V_new,  dtype=int)

    K = len(classes)
    idx = {c: i for i, c in enumerate(classes)}
    mat = np.zeros((K, K), dtype=int)

    # Only consider valid class values
    mask = np.isin(V_prev, classes) & np.isin(V_new, classes)
    vp = V_prev[mask]
    vn = V_new[mask]

    for a, b in zip(vp, vn):
        mat[idx[a], idx[b]] += 1

    # total flips = all off-diagonal transitions
    total_flips = int(mat.sum() - np.trace(mat))

    counts = {}
    for i, ca in enumerate(classes):
        for j, cb in enumerate(classes):
            if ca == cb:
                continue
            counts[f"{ca}→{cb}"] = int(mat[i, j])

    return total_flips, counts, mat

#endregion

#region AgentPy Helper Functions 
# This block provides routing, no-go checking, triage schedule, and safe-path helper functions for the AgentPy model.
def euclid(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])

def nearest_node(x, y):
    return min(nodes, key=lambda n: euclid((x, y),
        (G.nodes[n]["x"], G.nodes[n]["y"])))

def tile_class_at_time(model, xw, yw, t):
    """Return tile class (0/1/2) at world coordinate at time t."""
    # clamp t
    if t < 0:
        t = 0
    if hasattr(model, "tile_snapshots") and model.tile_snapshots:
        t = min(int(t), len(model.tile_snapshots) - 1)
        snap = model.tile_snapshots[t]   # shape = X.shape
    else:
        # fallback: try global tile_snapshots
        t = min(int(t), len(tile_snapshots) - 1) if 'tile_snapshots' in globals() else 0
        snap = tile_snapshots[t] if 'tile_snapshots' in globals() else None

    if snap is None:
        return 0  # fallback: treat as safe

    ix, iy = world_to_grid(xw, yw, x_min, y_min, Dx, Dy)

    if ix < 0 or iy < 0 or iy >= snap.shape[0] or ix >= snap.shape[1]:
        return 0  # outside map -> treat as safe (or change if you prefer)
    return int(snap[iy, ix])

def is_no_go_at_time(model, xw, yw, t):
    return tile_class_at_time(model, xw, yw, t) == int(NO_GO_TILE_CLASS)

def build_forbidden_nodes_at_time_uncached(model, t):
    """
    Mark graph nodes as forbidden when the node sits in a no-go tile at time t.
    Uses precomputed model._node_grid_ix/_node_grid_iy.
    """
    if not hasattr(model, "_node_grid_ix"):
        return set()

    # pick snapshot
    if hasattr(model, "tile_snapshots") and model.tile_snapshots:
        t = min(int(t), len(model.tile_snapshots) - 1)
        snap = model.tile_snapshots[t]
    else:
        t = min(int(t), len(tile_snapshots) - 1) if 'tile_snapshots' in globals() else 0
        snap = tile_snapshots[t] if 'tile_snapshots' in globals() else None

    if snap is None:
        return set()

    forbidden = set()
    for n in nodes:
        ix = model._node_grid_ix[n]
        iy = model._node_grid_iy[n]
        if ix is None or iy is None:
            continue
        if 0 <= iy < snap.shape[0] and 0 <= ix < snap.shape[1]:
            if int(snap[iy, ix]) == int(NO_GO_TILE_CLASS):
                forbidden.add(n)
    return forbidden

def build_forbidden_nodes_at_time(model, t):
    """Cached version (per tick) if model supports it."""
    if hasattr(model, "get_forbidden_nodes"):
        return model.get_forbidden_nodes(t)
    return build_forbidden_nodes_at_time_uncached(model, t)

def nearest_node_safe(model, x, y, forbidden_nodes):
    """Nearest node that is not forbidden."""
    best = None
    best_d = float("inf")
    for n in nodes:
        if n in forbidden_nodes:
            continue
        dx = G.nodes[n]["x"] - x
        dy = G.nodes[n]["y"] - y
        d = dx*dx + dy*dy
        if d < best_d:
            best_d = d
            best = n
    return best

def shortest_path_safe(model, src, tx, ty, t):
    """
    Network shortest path avoiding forbidden nodes (no-go).
    Returns:
      - list of nodes AFTER src (same as before) if path has >= 1 edge
      - [] if src == tgt (already at target node)
      - None if no safe path exists
    """
    forbidden = build_forbidden_nodes_at_time(model, t)

    if src in forbidden:
        return None


    tgt = nearest_approach_node_safe(model, tx, ty, forbidden, t, k=30, step_m=25.0)
    if tgt is None:
        print(f"[t={t}] tgt=None for (tx,ty)=({tx:.1f},{ty:.1f}) src={src}")
        return None



    if src == tgt:
        return []

    try:
        # PERF: use per-tick cached safe graph if available
        H = model.get_safe_graph(t) if hasattr(model, "get_safe_graph") else nx.subgraph_view(G, filter_node=lambda n: n not in forbidden)
        return nx.shortest_path(H, src, tgt, weight="travel_time")[1:]
    except nx.NetworkXNoPath:
        return None

def segment_crosses_no_go(model, x0, y0, x1, y1, t, step_m=50.0):
    """
    Sample along straight segment and return True if any sample point is in a no-go tile.
    step_m: sampling distance in world units (meters if your coords are meters).
    """
    dx = x1 - x0
    dy = y1 - y0
    dist = math.hypot(dx, dy)
    if dist <= 0:
        return False

    # number of samples (include endpoint)
    n = max(1, int(dist / step_m))
    for i in range(n + 1):
        a = i / n
        xs = x0 + a * dx
        ys = y0 + a * dy
        if is_no_go_at_time(model, xs, ys, t):
            return True
    return False

def nearest_approach_node_safe(model, tx, ty, forbidden_nodes, t, k=30, step_m=25.0):
    # Query the k nearest nodes using the KD-tree.
    dists, idxs = model._kdtree.query([tx, ty], k=k)
    if np.isscalar(idxs):
        idxs = [int(idxs)]

    for idx in idxs:
        n = int(model._node_ids[idx])
        if n in forbidden_nodes:
            continue
        x0, y0 = model._node_xy[idx]
        if not segment_crosses_no_go(model, x0, y0, tx, ty, int(t), step_m=step_m):
            return n
    return None

def get_edge_travel_time(u, v):
    data = G.get_edge_data(u, v)
    if data is None:
        return None

    # MultiGraph case: dict of keys -> attrdict
    if "travel_time" not in data and isinstance(next(iter(data.values()), None), dict):
        edge = min(data.values(), key=lambda d: float(d.get("travel_time", 0.0)))
        return float(edge["travel_time"])

    # Graph case: attrdict
    return float(data.get("travel_time", 0.0))

def dijkstra_to_targets(G, src, targets, weight="travel_time", cutoff=float("inf")):
    """
    Dijkstra die alleen afstanden naar 'targets' teruggeeft en stopt zodra alle targets gevonden zijn,
    of zodra cutoff bereikt is.
    """
    if not targets:
        return {}

    # distances
    dist = {src: 0.0}
    pq = [(0.0, src)]
    found = {}
    remaining = set(targets)

    while pq and remaining:
        d, u = heapq.heappop(pq)
        if d != dist.get(u, math.inf):
            continue
        if d > cutoff:
            break

        if u in remaining:
            found[u] = d
            remaining.remove(u)
            if not remaining:
                break

        for v, edata in G[u].items():

            # --- MultiGraph / MultiDiGraph: edata is dict(key -> attrdict)
            if "travel_time" not in edata and any(isinstance(val, dict) for val in edata.values()):
                best_w = math.inf
                for _k, attrdict in edata.items():
                    w = float(attrdict.get(weight, 1.0))
                    if w < best_w:
                        best_w = w
                w = best_w

            # --- Graph / DiGraph: edata is attrdict
            else:
                w = float(edata.get(weight, 1.0))

            nd = d + w
            if nd < dist.get(v, math.inf) and nd <= cutoff:
                dist[v] = nd
                heapq.heappush(pq, (nd, v))


    return found

def shortest_path_to_node_safe(model, src, tgt, t):
    forbidden = build_forbidden_nodes_at_time(model, t)
    if src in forbidden or tgt in forbidden:
        return None

    H = model.get_safe_graph(t) if hasattr(model, "get_safe_graph") else nx.subgraph_view(
        G, filter_node=lambda n: n not in forbidden
    )
    try:
        return nx.shortest_path(H, src, tgt, weight="travel_time")[1:]
    except nx.NetworkXNoPath:
        return None

def segment_crosses_no_go_over_time(model, x0, y0, x1, y1, t0, travel_time_min, step_m=50.0):
    """
    Check of een recht segment (x0,y0)->(x1,y1) ooit een no-go tile kruist
    gedurende de reis, rekening houdend met tile flips in de tijd.

    We nemen aan dat beweging lineair is en dat tiles per minuut (tick) kunnen flippen.
    We checken per minuut (integer t) + spatial sampling langs het segment.
    """
    if travel_time_min is None or travel_time_min <= 0:
        # No travel time means only checking t0.
        return segment_crosses_no_go(model, x0, y0, x1, y1, int(t0), step_m=step_m)

    t0i = int(math.floor(t0))
    t1i = int(math.floor(t0 + travel_time_min))

    # Number of time samples, using one sample per minute.
    for tt in range(t0i, t1i + 1):
        # Fraction of elapsed travel time, clamped to [0, 1].
        a = 0.0 if t1i == t0i else (tt - t0) / travel_time_min
        a = max(0.0, min(1.0, a))

        xt = x0 + a * (x1 - x0)
        yt = y0 + a * (y1 - y0)

        # Spatial sampling around that moment, using a conservative check.
        # This can be made faster by doing only a point check here.
        if is_no_go_at_time(model, xt, yt, tt):
            return True

    # Optional pure spatial check at the start time.
    # Useful when step_m or line sampling is already used.
    return False

def path_nodes_safe_over_time(model, path_nodes_including_src, t0, samples_per_edge=3):
    if not path_nodes_including_src or len(path_nodes_including_src) == 1:
        n0 = path_nodes_including_src[0]
        x0 = float(G.nodes[n0]["x"]); y0 = float(G.nodes[n0]["y"])
        return not is_no_go_at_time(model, x0, y0, int(t0))

    t = float(t0)
    for i in range(len(path_nodes_including_src) - 1):
        u = path_nodes_including_src[i]
        v = path_nodes_including_src[i+1]
        tt = get_edge_travel_time(u, v)
        if tt is None:
            return False

        x_u = float(G.nodes[u]["x"]); y_u = float(G.nodes[u]["y"])
        x_v = float(G.nodes[v]["x"]); y_v = float(G.nodes[v]["y"])

        # sample along the edge
        for s in range(samples_per_edge + 1):
            a = s / samples_per_edge
            ts = t + a * float(tt)
            xs = x_u + a * (x_v - x_u)
            ys = y_u + a * (y_v - y_u)
            if is_no_go_at_time(model, xs, ys, int(math.floor(ts))):
                return False

        t += float(tt)

    return True

def triage_at_abs_from_det_schedule(c, t_abs):
    tri0 = getattr(c, "triage0", getattr(c, "triage", "green"))
    sched = getattr(c, "det_schedule", []) or []
    label = str(tri0)
    for e in sched:
        if float(e.get("t", float("inf"))) <= float(t_abs):
            label = str(e.get("triage", label))
        else:
            break
    return label

def will_be_black_by_time(model, c, t_abs):
    # schedule-based (bed-independent)
    return triage_at_abs_from_det_schedule(c, t_abs) == "black"

def will_position_be_no_go(model, x, y, t_future):
    return is_no_go_at_time(model, x, y, int(t_future))

def nearest_node_safe_at_time(model, x, y, t_future):
    forbidden_future = build_forbidden_nodes_at_time(model, int(t_future))
    n = nearest_node_safe(model, x, y, forbidden_future)
    return n if n is not None else nearest_node(x, y)

def triage_from_schedule(c, t_abs):
    tri0 = getattr(c, "triage0", getattr(c, "triage", "green"))
    sched = getattr(c, "det_schedule", []) or []
    label = str(tri0)
    for e in sched:
        if float(e.get("t", float("inf"))) <= float(t_abs):
            label = str(e.get("triage", label))
        else:
            break
    return label
#endregion

#region AgentPy
# This block defines the AgentPy agents, coordinators, platforms, model setup, and simulation step logic.
class Casualty(ap.Agent):

    def setup(self):
        self.cid = int(self.model._cid_seq)
        self.model._cid_seq += 1

        self.evacuated = False
        self.assigned = False

        self.x = None
        self.y = None

        self.triage = sample_triage_label()
        self.current_triage = self.triage

        self.time_since_spawn = 0.0

        self.stabilization_time_min = sample_stabilization_time(self.triage)
        self.max_tolerable_delay_min = sample_max_delay(self.triage)

        self.time_to_next_progression = self.max_tolerable_delay_min

        self.t_created = float(self.model.t)        
        self.pickup_time = None                    
        self.arrival_base_t = None                 
        self.t_bed = None                           
        self.treatment_start_t = None
        self.treatment_end_t = None

        self.died_in_queue = False
        self.picked_by = None                       # "ambulance" / "helicopter"
        
        self.no_go = False


        self.triage_events = []  
        # event = {"t": time_abs, "triage": label, "stab": stabilization_time_min}
        # --- NEW: bed-independent deterioration schedule (absolute times) ---
        self.triage0 = str(self.triage)              # initial label at spawn
        t0 = float(self.t_created)

        # We assume a fixed progression chain via progress_triage_label():
        # green -> yellow -> red -> black (or similar)
        schedule = []

        cur = self.triage0
        t_abs = t0

        # IMPORTANT: we pre-sample the time-to-next for each stage
        # using your existing sample_max_delay(label)
        # (this is now independent of future bed availability)
        for _ in range(4):  # safety cap
            if cur == "black":
                break

            dt = float(sample_max_delay(cur))        # minutes until next progression
            t_abs = t_abs + dt

            nxt = str(progress_triage_label(cur))
            if nxt == cur:
                break

            schedule.append({"t": float(t_abs), "triage": nxt})
            cur = nxt

        self.det_schedule = schedule
        # ---------------------------------------------------------------


        # initial event
        self.triage_events.append({
            "t": float(self.model.t),
            "triage": str(self.current_triage),
            "stab": float(self.stabilization_time_min),
        })

    def step(self):
        st = getattr(self, "state", None)
        if st in ("treatment", "stabilized", "dead", "dead_at_base"):
            return

        # update triage deterministically from schedule
        new_label = triage_from_schedule(self, float(self.model.t))

        if new_label != self.current_triage:
            self.current_triage = new_label

            # Optionally update stabilization time when triage changes.
            self.stabilization_time_min = sample_stabilization_time(new_label)

            # log event
            self.triage_events.append({
                "t": float(self.model.t),
                "triage": str(self.current_triage),
                "stab": float(self.stabilization_time_min),
            })

class Base(ap.Agent):
    def setup(self):
        # Position 
        self.x = float(self.model.base_x)
        self.y = float(self.model.base_y)
        self.node = self.model.base_node

        # Base capaciteit en policy 
        self.n_beds = int(self.model.N_BEDS)
        self.queue_policy = str(self.model.BASE_POLICY)  # "FIFO" of "SEVERITY"

        self.queue = []
        self.in_treatment = []   # list of [casualty, remaining_time_min]

        self.queue_triage_changes = {
            "green→yellow": 0,
            "yellow→red": 0,
            "red→black": 0
        }

        self.history = {
            "t": [],
            "queue_total": [],
            "queue_by_triage": [],
            "beds_total": [],
            "beds_by_triage": [],
            "queue_triage_changes": []
        }

        self.queue_lengths = []
        self.max_queue_length = 0

    # --------------------------------------------------
    def admit(self, c, t):
        """
        Called when a platform drops a casualty at the base.
        - Records arrival time
        - Logs arrival for base replay / metrics
        - If black on arrival -> dead_at_base (NOT queued)
        - Else -> enter queue
        """
        # ---- 1) Always record arrival time ----
        c.arrival_base_t = float(t)

        # ---- 2) Log arrival for replay/metrics ----
        # Make sure base_arrivals list exists
        if not hasattr(self.model, "base_arrivals") or self.model.base_arrivals is None:
            self.model.base_arrivals = []

        self.model.base_arrivals.append({
            "cid": int(getattr(c, "cid", -1)),
            "t_arrival": float(t),
            "t_created": float(getattr(c, "t_created", t)),
            "pickup_time": None if getattr(c, "pickup_time", None) is None else float(c.pickup_time),
            "picked_by": None if getattr(c, "picked_by", None) is None else str(c.picked_by),
            "no_go": bool(getattr(c, "no_go", False)),

            # realized triage event log (if you keep it)
            "triage_events": list(getattr(c, "triage_events", [])),

            # bed-independent deterioration schedule (if you added it)
            "triage0": str(getattr(c, "triage0", getattr(c, "triage", "green"))),
            "det_schedule": list(getattr(c, "det_schedule", [])),
        })

        # ---- 3) If black on arrival: dead at base, not queue death ----
        if getattr(c, "current_triage", None) == "black":
            # mission aborted reason: black on arrival
            self.model.record_mission_abort(
                kind="black_on_arrival",
                platform_type=getattr(c, "picked_by", None),
                agent_id=None,
                casualty=c,
                t=t,
            )
            c.state = "dead_at_base"
            c.died_in_queue = False
            return

        # ---- 4) Otherwise: enter queue ----
        c.state = "queue"
        self.queue.append(c)


    # --------------------------------------------------
    def _select_from_queue(self):
        # Remove any black-label casualties from the queue first as a failsafe.
        self.queue = [c for c in self.queue if getattr(c, "current_triage", None) != "black"]

        if not self.queue:
            return None

        if self.queue_policy.upper() == "FIFO":
            return self.queue.pop(0)

        # Severity-based selection, prioritizing red first.
        severity = {"red": 1, "yellow": 2, "green": 3}
        c = min(self.queue, key=lambda c: severity.get(getattr(c, "current_triage", "green"), 3))
        self.queue.remove(c)
        return c

    # --------------------------------------------------
    def step(self):
        TIME_STEP_MINUTES = float(self.model.TIMESTEP_DURATION)

        # ---------------------------------
        # Metrics patch: died in queue
        # ---------------------------------
        still_waiting = []
        for c in self.queue:
            if getattr(c, "current_triage", None) == "black":
                c.died_in_queue = True
                c.state = "dead"
            else:
                still_waiting.append(c)
        self.queue = still_waiting

        # queue length tracking
        q_len = len(self.queue)
        self.queue_lengths.append(q_len)
        self.max_queue_length = max(self.max_queue_length, q_len)

        finished = []
        for entry in self.in_treatment:
            entry[1] -= TIME_STEP_MINUTES
            if entry[1] <= 0:
                finished.append(entry)

        # finish treatment
        for c, _ in finished:
            self.in_treatment = [e for e in self.in_treatment if e[0] != c]
            c.state = "stabilized"
            c.treatment_end_t = self.model.t

        while len(self.in_treatment) < self.n_beds and self.queue:
            c = self._select_from_queue()
            if c is None:
                break

            if getattr(c, "current_triage", None) == "black":
                c.state = "dead"
                c.died_in_queue = True
                continue

            c.state = "treatment"
            c.treatment_start_t = self.model.t
            c.t_bed = self.model.t
            self.in_treatment.append([c, float(getattr(c, "stabilization_time_min", 0.0))])

        # Logging
        triage_labels = ["green", "yellow", "red", "black"]

        queue_counter = Counter([getattr(c, "current_triage", "green") for c in self.queue])
        beds_counter = Counter([getattr(c, "current_triage", "green") for c, _ in self.in_treatment])

        self.history["t"].append(self.model.t)
        self.history["queue_total"].append(len(self.queue))
        self.history["queue_by_triage"].append({k: queue_counter.get(k, 0) for k in triage_labels})

        self.history["beds_total"].append(len(self.in_treatment))
        self.history["beds_by_triage"].append({k: beds_counter.get(k, 0) for k in ["green", "yellow", "red"]})

        self.history["queue_triage_changes"].append(dict(self.queue_triage_changes))

class Coordinator(ap.Agent):
    """
    Central coordinator that handles dispatch and task assignment.

    Heuristic 1:
    - Ambulance: severity (red>yellow>green) then shortest safe travel time (from ambulance)
    - Helicopter: severity then MAX ambulance safe travel time from base (hard-to-reach for ambulances)

    Heuristic 2:
    - Ambulance: shortest safe travel time only (from ambulance)
    - Helicopter: MAX ambulance safe travel time from base only

    Heuristic 3:
    - Same target-selection rule as Heuristic 1
    - Dispatch order: helicopters first, then ambulances

    Heuristic 4:
    - Same target-selection rule as Heuristic 2
    - Dispatch order: helicopters first, then ambulances

    Heuristic 5:
    - Ambulance: shortest safe travel time only (closest to the ambulance)
    - Helicopter: severity then MAX ambulance safe travel time from base

    Heuristic 6:
     - Same target-selection rule as Heuristic 5
     - Dispatch order: helicopters first, then ambulances


    Black-label casualties are never dispatched to ambulances or helicopters.
    """


    # -------------------------
    # Setup + caches
    # -------------------------
    def setup(self):
        # candidates cache
        self._cand_cache_t = None
        self._cand_cache = []

        # base->casualty travel time cache
        self._amb_time_cache_t = None
        self._amb_time_cache = {}

        # Helicopter reachability cache, used when enabled.
        self._heli_reach_cache_t = None
        self._heli_reach_cache = {}

        # Initialize the base-distance cache used by _get_base_dist.
        self._base_dist_t = None
        self._base_dist = None

        # Per-tick cache for single-source Dijkstra results.
        self._sssp_cache_t = None
        self._sssp_cache = {}   # key: (src_node, cutoff) -> dist-dict

        # Per-tick cache for approach nodes per casualty.
        self._approach_cache_t = None
        self._approach_cache = {}  # key: id(c) -> tgt_node



    def _get_sssp_dist(self, m, src_node):
        tt = int(m.t)
        if self._sssp_cache_t != tt:
            self._sssp_cache_t = tt
            self._sssp_cache = {}

        if src_node in self._sssp_cache:
            return self._sssp_cache[src_node]

        H = m.get_safe_graph(tt)
        cutoff = float(getattr(m, "DIJKSTRA_CUTOFF", 60.0))
        dist = nx.single_source_dijkstra_path_length(H, src_node, cutoff=cutoff, weight="travel_time")

        self._sssp_cache[src_node] = dist
        return dist

    def _get_approach_node(self, m, c, forbidden, ttick):
        """Return approach node near casualty, cached per tick."""
        if self._approach_cache_t != ttick:
            self._approach_cache_t = ttick
            self._approach_cache = {}

        key = id(c)
        if key in self._approach_cache:
            return self._approach_cache[key]

        tgt = nearest_approach_node_safe(
            m, c.x, c.y, forbidden, ttick,
            k=30, step_m=25.0
        )
        self._approach_cache[key] = tgt
        return tgt

    # -------------------------
    # Severity ranking
    # -------------------------
    def _severity_rank(self, triage):
        # lower = more severe
        return {"red": 0, "yellow": 1, "green": 2}.get(triage, 2)

    # -------------------------
    # Candidates (1x per tick)
    # -------------------------
    def _available_candidates(self, m):
        """
        Available casualties:
        - not evacuated
        - not assigned
        - NOT black
        - casualty tile not in no-go at current time
        """
        t = int(m.t)
        out = []
        for c in m.casualties:
            if getattr(c, "evacuated", False) or getattr(c, "assigned", False):
                continue

            if getattr(c, "current_triage", None) == "black":
                continue

            if is_no_go_at_time(m, c.x, c.y, t):
                c.no_go = True

                # unreachable: filtered because current tile is no-go
                m.unreachable["filtered_no_go_current_freq"] += 1
                if not getattr(c, "_flag_filtered_no_go", False):
                    c._flag_filtered_no_go = True
                    m.unreachable["filtered_no_go_current_unique"] += 1

                continue


            c.no_go = False
            out.append(c)
        return out

    # -------------------------
    # SAFE travel time only (no path build)
    # -------------------------
    def _safe_travel_time(self, m, src_node, tx, ty, t, k_approach=30, step_m=25.0):
        """
        Fast safe travel time using bidirectional_dijkstra (usually faster than dijkstra_path_length).
        Returns float or None.
        """
        tt = int(t)
        forbidden = m.get_forbidden_nodes(tt) if hasattr(m, "get_forbidden_nodes") else set()
        if src_node in forbidden:
            return None

        tgt = nearest_approach_node_safe(m, tx, ty, forbidden, tt, k=k_approach, step_m=step_m)
        if tgt is None:
            return None

        H = m.get_safe_graph(tt) if hasattr(m, "get_safe_graph") else G
        try:
            dist, _path = nx.bidirectional_dijkstra(H, src_node, tgt, weight="travel_time")
            return dist
        except nx.NetworkXNoPath:
            return None

        # -------------------------
        # Prefilter (Top-M Euclid) to reduce Dijkstra calls
        # -------------------------

    def _prefilter_candidates_for_heli(self, candidates, m, prefer="far"):
        """
        Prefilter om Dijkstra/line-of-sight checks te beperken.

        prefer="far": neem M verst (euclid) van base (proxy voor hoge amb travel time)
        prefer="near": neem M dichtst (euclid) bij base
        """
        M = int(getattr(m, "HELI_PREFILTER_M", 10))
        if len(candidates) <= M:
            return candidates

        bx, by = m.base_x, m.base_y

        if prefer == "near":
            return heapq.nsmallest(
                M,
                candidates,
                key=lambda c: (c.x - bx) ** 2 + (c.y - by) ** 2
            )
        else:
            return heapq.nlargest(
                M,
                candidates,
                key=lambda c: (c.x - bx) ** 2 + (c.y - by) ** 2
            )

    def _ambulance_best_target(self, a, candidates, m, heuristic_id, claimed):
        if not candidates:
            return None, None, None

        ttick = int(m.t)
        forbidden = m.get_forbidden_nodes(ttick)
        if a.node in forbidden:
            return None, None, None

        # Filter assigned/claimed
        candidates = [
            c for c in candidates
            if (not getattr(c, "assigned", False)) and (id(c) not in claimed)
        ]
        if not candidates:
            return None, None, None

        # Prefilter M closest by euclid
        M = int(getattr(m, "PREFILTER_M", 12))
        if len(candidates) > M:
            ax, ay = a.x, a.y
            cand = heapq.nsmallest(M, candidates, key=lambda c: (c.x - ax) ** 2 + (c.y - ay) ** 2)
        else:
            cand = candidates

        # Build approach targets
        cand_info = []
        targets = []
        for c in cand:
            tgt = nearest_approach_node_safe(m, c.x, c.y, forbidden, ttick, k=30, step_m=25.0)
            if tgt is None:
                # unreachable: no approach node
                m.unreachable["amb_no_approach_freq"] += 1
                if not getattr(c, "_flag_amb_no_approach", False):
                    c._flag_amb_no_approach = True
                    m.unreachable["amb_no_approach_unique"] += 1
                continue

            cand_info.append((c, tgt))
            targets.append(tgt)

        if not cand_info:
            return None, None, None

        # 1x multi-target dijkstra
        H = m.get_safe_graph(ttick)
        cutoff = float(getattr(m, "DIJKSTRA_CUTOFF", 60.0))
        target_dists = dijkstra_to_targets(H, a.node, targets, weight="travel_time", cutoff=cutoff)

        # IMPORTANT: define these BEFORE using them
        best = None
        best_tt = None
        best_key = None

        for c, tgt in cand_info:
            # failsafe: skip if in the meantime claimed/assigned
            if getattr(c, "assigned", False) or (id(c) in claimed):
                continue

            d = target_dists.get(tgt, None)
            if d is None:
                # unreachable: no safe path (or not found within cutoff)
                m.unreachable["amb_no_path_freq"] += 1
                if not getattr(c, "_flag_amb_no_path", False):
                    c._flag_amb_no_path = True
                    m.unreachable["amb_no_path_unique"] += 1
                continue


            if heuristic_id in (2, 5, 6):
                 key = (d,)
            else:
                # Heuristic 1,3: severity first, then distance
                key = (self._severity_rank(getattr(c, "current_triage", "green")), d)

            if (best is None) or (key < best_key):
                best = c
                best_tt = d
                best_key = key

        if best is None:
            return None, None, None

        chosen_path = shortest_path_safe(m, a.node, best.x, best.y, m.t)
        if chosen_path is None:
            return None, None, None

        return best, chosen_path, best_tt

    def _get_base_dist(self, m):
        tt = int(m.t)
        if self._base_dist_t != tt:
            self._base_dist_t = tt
            H = m.get_safe_graph(tt)
            cutoff = float(getattr(m, "DIJKSTRA_CUTOFF", 60.0))
            self._base_dist = nx.single_source_dijkstra_path_length(
                H, m.base_node, cutoff=cutoff, weight="travel_time"
            )
        return self._base_dist

    def _get_amb_time_cache(self, m):
            tt = int(m.t)
            if self._amb_time_cache_t != tt:
                self._amb_time_cache_t = tt
                self._amb_time_cache = {}
            return self._amb_time_cache

    def _amb_reach_time_from_base(self, c, m):
        cache = self._get_amb_time_cache(m)
        key = id(c)
        if key in cache:
            return cache[key]

        ttick = int(m.t)
        forbidden = m.get_forbidden_nodes(ttick)

        tgt = nearest_approach_node_safe(m, c.x, c.y, forbidden, ttick, k=30, step_m=25.0)
        if tgt is None:
            cache[key] = None
            return None

        base_dist = self._get_base_dist(m)
        val = base_dist.get(tgt, None)
        cache[key] = val
        return val

    def _heli_reachable_straight(self, h, c, m):
            """
            Heli constraint:
            - heli position not in no-go
            - straight line heli->casualty must not cross no-go
            - casualty itself is already filtered by _available_candidates (not no-go, not black)
            """
            t = int(m.t)

            if is_no_go_at_time(m, h.x, h.y, t):
                return False

            # If you prefer, you can relax this check for speed (bigger step_m)
            step_m = float(getattr(m, "HELI_STEP_M", 120.0))
            if segment_crosses_no_go(m, h.x, h.y, c.x, c.y, t, step_m=step_m):
                # unreachable for heli: straight line blocked by no-go
                m.unreachable["heli_line_blocked_freq"] += 1
                if not getattr(c, "_flag_heli_line_blocked", False):
                    c._flag_heli_line_blocked = True
                    m.unreachable["heli_line_blocked_unique"] += 1
                return False


            return True

        # -------------------------
        # Helicopter: choose best target
        # -------------------------

    def _heli_reachable_cached(self, h, c, m):
        tt = int(m.t)
        if self._heli_reach_cache_t != tt:
            self._heli_reach_cache_t = tt
            self._heli_reach_cache = {}

        key = id(c)
        if key in self._heli_reach_cache:
            return self._heli_reach_cache[key]

        ok = self._heli_reachable_straight(h, c, m)
        self._heli_reach_cache[key] = ok
        return ok

    def _dispatch_ambulances(self, m, candidates, claimed):
        heuristic_id = int(getattr(m, "DISPATCH_HEURISTIC", 1))
        forbidden = m.get_forbidden_nodes(m.t) if hasattr(m, "get_forbidden_nodes") else set()

        # Local copy that is consumed during assignment.
        available = [
            c for c in candidates
            if (not getattr(c, "assigned", False)) and (id(c) not in claimed)
        ]

        for a in m.ambulances:
            if getattr(a, "disabled", False):
                continue
            if a.busy or a.offroad or a.returning or a.picking_up:
                continue
            if getattr(a, "node", None) in forbidden:
                continue
            if len(a.cargo) >= a.capacity:
                a._start_return_to_base()
                continue

            if not available:
                if len(a.cargo) > 0:
                    a._start_return_to_base()
                continue

            chosen, chosen_path, _ = self._ambulance_best_target(a, available, m, heuristic_id, claimed)

            if chosen is None:
                if len(a.cargo) > 0:
                    a._start_return_to_base()
                continue

            # Claim and remove the casualty to prevent duplicate assignment.
            claimed.add(id(chosen))
            chosen.assigned = True
            if chosen in available:
                available.remove(chosen)

            a.target = chosen
            if len(a.cargo) > 0:
                a.return_after_next_pickup = True
            a.busy = True
            a.needs_route = True

            # Important: set the route immediately to avoid replanning in Ambulance.step.
            # This also ensures the selected path is actually used.
            a.path = chosen_path if chosen_path is not None else []
            a.needs_route = False  # a route has already been assigned

            a.current_edge = None
            a.edge_time_left = 0.0

    def _heli_best_target(self, h, candidates, m, heuristic_id, claimed):
        """
        Helicopter target selection:
        - Uses ambulance safe travel time from base as proxy.
        - "Farthest for ambulances from base" = MAX amb travel time from base.

        Mapping for helicopter target selection:
        1: severity then MAX amb_time_from_base
        2: MAX amb_time_from_base only
        3: zelfde als 1 (dispatch order differs)
        4: zelfde als 2 (dispatch order differs)
        5: same helicopter target selection as 1 (ambulance rule differs)
        6: same helicopter target selection as 5 (dispatch order differs: helicopters first)
        """
        if not candidates:
            return None

        # Map heuristics for heli behavior
        # Heuristic 6 must be identical to 5 for target selection (severity then MAX amb_time_from_base)
        if heuristic_id in (3, 5, 6):
            hid = 1
        elif heuristic_id in (4,):
            hid = 2
        else:
            hid = heuristic_id  # 1 or 2

        cand = [
            c for c in candidates
            if (not getattr(c, "assigned", False)) and (id(c) not in claimed)
        ]
        if not cand:
            return None

        # keep far-by-euclid prefilter (cheap proxy)
        cand = self._prefilter_candidates_for_heli(cand, m)

        reachable = [c for c in cand if self._heli_reachable_cached(h, c, m)]
        if not reachable:
            return None

        best = None
        best_key = None

        for c in reachable:
            amb_t = self._amb_reach_time_from_base(c, m)
            if amb_t is None:
                continue

            if hid == 2:
                # maximize amb_t
                key = (-amb_t,)
            else:
                # severity first, then maximize amb_t
                sev = self._severity_rank(getattr(c, "current_triage", "green"))
                key = (sev, -amb_t)

            if (best is None) or (key < best_key):
                best = c
                best_key = key

        return best

    def _dispatch_helicopters(self, m, candidates, claimed):
        heuristic_id = int(getattr(m, "DISPATCH_HEURISTIC", 1))

        # Local copy that is consumed during assignment.
        available = [
            c for c in candidates
            if (not getattr(c, "assigned", False)) and (id(c) not in claimed)
        ]

        for h in m.helicopters:
            if getattr(h, "disabled", False):
                continue
            # if full -> go home
            if (not h.returning) and (len(h.cargo) >= h.capacity):
                h.returning = True
                h.busy = False
                if h.target is not None:
                    h.target.assigned = False
                h.target = None

            # skip if not available for new mission
            if h.returning or h.busy or h.picking_up or (len(h.cargo) >= h.capacity):
                continue

            if not available:
                if len(h.cargo) > 0:
                    h.returning = True
                continue

            chosen = self._heli_best_target(h, available, m, heuristic_id, claimed)
            if chosen is None:
                if len(h.cargo) > 0:
                    h.returning = True
                continue

            # claim
            claimed.add(id(chosen))
            chosen.assigned = True
            if chosen in available:
                available.remove(chosen)

            h.target = chosen
            h.busy = True

    def step(self):
        m = self.model
        tt = int(m.t)

        if self._cand_cache_t != tt:
            self._cand_cache_t = tt
            self._cand_cache = self._available_candidates(m)

        candidates = self._cand_cache

        # per tick “voertuigen praten met elkaar”
        claimed = set()

        heuristic_id = int(getattr(m, "DISPATCH_HEURISTIC", 1))

        # Heuristics 3, 4, and 6 dispatch helicopters first, then ambulances.
        if heuristic_id in (3, 4, 6):
            self._dispatch_helicopters(m, candidates, claimed)
            self._dispatch_ambulances(m, candidates, claimed)
        else:
            # Default dispatch order: ambulances first, then helicopters.
            self._dispatch_ambulances(m, candidates, claimed)
            self._dispatch_helicopters(m, candidates, claimed)

class ClairvoyantCoordinator(Coordinator):
    """
    Coordinator that looks ahead in tile_snapshots and only dispatches when:
      - the casualty is not black-label now or at pickup completion, using a tick-conservative check
      - the route does not pass through current or future no-go tiles, including network and off-road travel
      - optionally, the casualty can also arrive at the base before the black-label deadline

    Drop-in replacement for the standard ClairvoyantCoordinator.
    """

    def setup(self):
        super().setup()
        self._feas_cache_t = None
        self._feas_cache = {}  # key: (platform_type, platform_id, casualty_id) -> bool

    # -------------------------
    # Bed-independent triage via det_schedule
    # -------------------------
    def triage_at_abs(self, c, t_abs):
        tri0 = getattr(c, "triage0", getattr(c, "triage", "green"))
        sched = getattr(c, "det_schedule", []) or []
        label = str(tri0)
        for e in sched:
            if float(e.get("t", float("inf"))) <= float(t_abs):
                label = str(e.get("triage", label))
            else:
                break
        return label

    def black_time_abs(self, c):
        sched = getattr(c, "det_schedule", []) or []
        for e in sched:
            if str(e.get("triage")) == "black":
                return float(e.get("t"))
        return float("inf")

    def _deadline_for_casualty(self, c, m):
        # Deadline is the time when the casualty becomes black-label, independent of bed availability.
        return self.black_time_abs(c)

    def _will_be_black_tick_conservative(self, c, t_abs):
        """
        Conservative check for discrete ticks:
        check both floor and ceiling of t_abs.
        """
        t0 = int(math.floor(t_abs))
        t1 = int(math.ceil(t_abs))
        return (self.triage_at_abs(c, t0) == "black") or (self.triage_at_abs(c, t1) == "black")

    # -------------------------
    # Proactive handling to avoid exogenous platform failures when a tile flips under a vehicle.
    # -------------------------
    def _proactive_avoid_future_no_go(self, m):
        tt = int(m.t)
        t_next = tt + 1

        # --- Ambulances ---
        for a in m.ambulances:
            if getattr(a, "disabled", False):
                continue

            if is_no_go_at_time(m, a.x, a.y, tt) or will_position_be_no_go(m, a.x, a.y, t_next):

                if a.target is not None:
                    a.target.assigned = False

                safe_n = nearest_node_safe_at_time(m, a.x, a.y, t_next)

                a.x = float(G.nodes[safe_n]["x"])
                a.y = float(G.nodes[safe_n]["y"])
                a.node = safe_n

                a.busy = False
                a.target = None
                a.path = []
                a.current_edge = None
                a.edge_time_left = 0.0
                a.needs_route = False

                a.picking_up = False
                a.pickup_timer = 0.0
                a.pickup_target = None

                a.offroad = False
                a.offroad_mode = None
                a.reentry_node = None

                if len(a.cargo) > 0:
                    a.returning = True
                else:
                    a.returning = False
                    a.return_path = []
                    a.return_edge = None
                    a.return_edge_time_left = 0.0

        # --- Helicopters ---
        for h in m.helicopters:
            if getattr(h, "disabled", False):
                continue

            if is_no_go_at_time(m, h.x, h.y, tt) or will_position_be_no_go(m, h.x, h.y, t_next):

                if h.target is not None:
                    h.target.assigned = False

                h.x = float(m.base_x)
                h.y = float(m.base_y)

                h.busy = False
                h.target = None

                h.picking_up = False
                h.pickup_timer = 0.0
                h.pickup_target = None

                h.returning = False

    # -------------------------
    # Feasibility checks
    # -------------------------
    def _ambulance_can_complete(self, a, c, m, ttick, approach_node, travel_to_approach_min):
        """
        Clairvoyant check voor ambulance:
          - veilig netwerk tot approach (met safe-over-time check)
          - veilig offroad segment in de tijd
          - casualty niet black bij pickup completion (tick-conservatief)
          - casualty tile niet no-go bij pickup completion
          - (optioneel) ook base arrival vóór black-time
        """
        if self._feas_cache_t != ttick:
            self._feas_cache_t = ttick
            self._feas_cache = {}

        key = ("ambulance", getattr(a, "id", id(a)), id(c))
        if key in self._feas_cache:
            return self._feas_cache[key]

        if approach_node is None or travel_to_approach_min is None:
            self._feas_cache[key] = False
            return False

        # deadline = black-time
        dl = float(self._deadline_for_casualty(c, m))

        # 1) netwerkpad naar approach node
        path_to = shortest_path_to_node_safe(m, a.node, approach_node, ttick)
        if path_to is None:
            self._feas_cache[key] = False
            return False

        full_path_to = [a.node] + list(path_to)

        # Use the more conservative sampled path check when available; otherwise use node-only checking.
        if "path_safe_over_time_with_samples" in globals():
            ok_path = path_nodes_safe_over_time(m, full_path_to, ttick, samples_per_edge=3)
        else:
            ok_path = path_nodes_safe_over_time(m, full_path_to, ttick)

        if not ok_path:
            self._feas_cache[key] = False
            return False

        t_arrive_approach = float(ttick) + float(travel_to_approach_min)

        # 2) offroad naar casualty
        ax = float(G.nodes[approach_node]["x"])
        ay = float(G.nodes[approach_node]["y"])
        offroad_dist = euclid((ax, ay), (c.x, c.y))
        offroad_time = offroad_dist / float(m.OFFROAD_SPEED) if offroad_dist > 0 else 0.0

        if segment_crosses_no_go_over_time(m, ax, ay, c.x, c.y, t_arrive_approach, offroad_time, step_m=50.0):
            self._feas_cache[key] = False
            return False

        t_arrive_cas = t_arrive_approach + offroad_time

        # 3) pickup delay
        pickup_delay = float(getattr(m, "AMBULANCE_PICKUP_DELAY_MIN", 1.0))
        t_pickup_done = t_arrive_cas + pickup_delay

        # 3b) The casualty must not be black-label at pickup completion, using a tick-conservative check.
        if self._will_be_black_tick_conservative(c, t_pickup_done):
            self._feas_cache[key] = False
            return False

        # The casualty tile must still be safe at the pickup completion tick.
        if is_no_go_at_time(m, c.x, c.y, int(math.floor(t_pickup_done))):
            self._feas_cache[key] = False
            return False

        # 4) pickup moet vóór black-time klaar zijn
        if t_pickup_done > dl:
            self._feas_cache[key] = False
            return False

        # 5) Optionally require arrival at base before the casualty becomes black-label.
        require_base_before_black = bool(getattr(m, "CLAIRVOYANT_REQUIRE_BASE_BEFORE_BLACK", True))
        if require_base_before_black:
            # Off-road return to the approach node, conservatively using the same node.
            offroad_back_time = offroad_time  # zelfde dist / OFFROAD_SPEED als heen

            # Check that the off-road return segment does not cross future no-go tiles.
            if segment_crosses_no_go_over_time(
                m, c.x, c.y, ax, ay,
                t_pickup_done, offroad_back_time, step_m=50.0
            ):
                self._feas_cache[key] = False
                return False

            t_back_on_network = t_pickup_done + offroad_back_time
            t_after = int(math.floor(t_back_on_network))

            path_back = shortest_path_to_node_safe(m, approach_node, m.base_node, t_after)
            if path_back is None:
                self._feas_cache[key] = False
                return False

            # Network travel time back to base.
            t_est = float(t_back_on_network)
            u = approach_node
            for v in path_back:
                tt_edge = get_edge_travel_time(u, v)
                if tt_edge is None:
                    self._feas_cache[key] = False
                    return False
                t_est += float(tt_edge)
                u = v

            if dl is not None and t_est > float(dl):
                self._feas_cache[key] = False
                return False

        # --- Estimate: pickup -> back to network -> base (so we can validate ALL cargo deadlines) ---
        # 1) determine a realistic re-entry node at pickup completion time (nearest SAFE node then)
        t_pick = int(math.floor(t_pickup_done))
        forbidden_pick = m.get_forbidden_nodes(t_pick)
        reentry = nearest_node_safe(m, c.x, c.y, forbidden_pick)
        if reentry is None:
            self._feas_cache[key] = False
            return False

        rx = float(G.nodes[reentry]["x"])
        ry = float(G.nodes[reentry]["y"])

        # 2) offroad from casualty location back to reentry
        offroad_back_dist = euclid((c.x, c.y), (rx, ry))
        offroad_back_time = offroad_back_dist / float(m.OFFROAD_SPEED) if offroad_back_dist > 0 else 0.0

        # no-go-over-time check for that offroad return segment
        if segment_crosses_no_go_over_time(m, c.x, c.y, rx, ry, t_pickup_done, offroad_back_time, step_m=50.0):
            self._feas_cache[key] = False
            return False

        t_back_on_network = t_pickup_done + offroad_back_time
        t_after = int(math.floor(t_back_on_network))

        # 3) network reentry -> base
        path_back = shortest_path_to_node_safe(m, reentry, m.base_node, t_after)
        if path_back is None:
            self._feas_cache[key] = False
            return False

        t_est = float(t_back_on_network)
        u = reentry
        for v in path_back:
            tt_edge = get_edge_travel_time(u, v)
            if tt_edge is None:
                self._feas_cache[key] = False
                return False
            t_est += float(tt_edge)
            u = v

        # 4) Deadline check for EVERYONE in cargo + this candidate
        # Use bed-independent black-time schedule from the coordinator:
        def black_time_abs_local(cas):
            sched = getattr(cas, "det_schedule", []) or []
            for e in sched:
                if str(e.get("triage")) == "black":
                    return float(e.get("t"))
            return float("inf")

        # check new casualty
        if t_est > black_time_abs_local(c):
            self._feas_cache[key] = False
            return False

        # check existing cargo casualties on board
        for cc in getattr(a, "cargo", []):
            if t_est > black_time_abs_local(cc):
                self._feas_cache[key] = False
                return False

        self._feas_cache[key] = True
        return True

    def _heli_can_complete(self, h, c, m, ttick):
        """
        Heli:
          now -> casualty (straight)
          pickup delay
          casualty -> base (straight)
        met no-go over tijd + black tick-conservatief.
        """
        if self._feas_cache_t != ttick:
            self._feas_cache_t = ttick
            self._feas_cache = {}

        key = ("helicopter", getattr(h, "id", id(h)), id(c))
        if key in self._feas_cache:
            return self._feas_cache[key]

        if is_no_go_at_time(m, h.x, h.y, ttick):
            self._feas_cache[key] = False
            return False

        dl = float(self._deadline_for_casualty(c, m))

        # heen
        d1 = euclid((h.x, h.y), (c.x, c.y))
        t1 = d1 / float(m.HELI_SPEED) if d1 > 0 else 0.0

        if segment_crosses_no_go_over_time(
            m, h.x, h.y, c.x, c.y, ttick, t1,
            step_m=float(getattr(m, "HELI_STEP_M", 120.0))
        ):
            self._feas_cache[key] = False
            return False

        t_arrive = float(ttick) + t1

        pickup_delay = float(getattr(m, "HELICOPTER_PICKUP_DELAY_MIN", 3.0))
        t_pickup_done = t_arrive + pickup_delay

        if self._will_be_black_tick_conservative(c, t_pickup_done):
            self._feas_cache[key] = False
            return False
        if is_no_go_at_time(m, c.x, c.y, int(math.floor(t_pickup_done))):
            self._feas_cache[key] = False
            return False

        # Return path.
        d2 = euclid((c.x, c.y), (m.base_x, m.base_y))
        t2 = d2 / float(m.HELI_SPEED) if d2 > 0 else 0.0

        if segment_crosses_no_go_over_time(
            m, c.x, c.y, m.base_x, m.base_y, t_pickup_done, t2,
            step_m=float(getattr(m, "HELI_STEP_M", 120.0))
        ):
            self._feas_cache[key] = False
            return False

        t_base = t_pickup_done + t2

        # base vóór black-time
        if t_base > dl:
            self._feas_cache[key] = False
            return False

        self._feas_cache[key] = True
        return True

    # -------------------------
    # Override: ambulance target keuze met feasibility filter
    # -------------------------
    def _ambulance_best_target(self, a, candidates, m, heuristic_id, claimed):
        if not candidates:
            return None, None, None

        ttick = int(m.t)
        forbidden = m.get_forbidden_nodes(ttick)
        if a.node in forbidden:
            return None, None, None

        candidates = [
            c for c in candidates
            if (not getattr(c, "assigned", False)) and (id(c) not in claimed)
        ]
        if not candidates:
            return None, None, None

        # Prefilter M closest by euclid
        M = int(getattr(m, "PREFILTER_M", 12))
        if len(candidates) > M:
            ax, ay = a.x, a.y
            cand = heapq.nsmallest(M, candidates, key=lambda c: (c.x - ax) ** 2 + (c.y - ay) ** 2)
        else:
            cand = candidates

        cand_info = []
        targets = []
        for c in cand:
            tgt = nearest_approach_node_safe(m, c.x, c.y, forbidden, ttick, k=30, step_m=25.0)
            if tgt is None:
                # unreachable: no approach node
                m.unreachable["amb_no_approach_freq"] += 1
                if not getattr(c, "_flag_amb_no_approach", False):
                    c._flag_amb_no_approach = True
                    m.unreachable["amb_no_approach_unique"] += 1
                continue

            cand_info.append((c, tgt))
            targets.append(tgt)


        if not cand_info:
            return None, None, None

        # 1x multi-target dijkstra
        H = m.get_safe_graph(ttick)
        cutoff = float(getattr(m, "DIJKSTRA_CUTOFF", 60.0))
        target_dists = dijkstra_to_targets(H, a.node, targets, weight="travel_time", cutoff=cutoff)

        best = None
        best_tt = None
        best_key = None

        for c, tgt in cand_info:
            if getattr(c, "assigned", False) or (id(c) in claimed):
                continue

            d = target_dists.get(tgt, None)
            if d is None:
                # unreachable: no safe path (or not found within cutoff)
                m.unreachable["amb_no_path_freq"] += 1
                if not getattr(c, "_flag_amb_no_path", False):
                    c._flag_amb_no_path = True
                    m.unreachable["amb_no_path_unique"] += 1
                continue


            # clairvoyant feasibility
            if not self._ambulance_can_complete(a, c, m, ttick, tgt, d):
                continue

            # Heuristic 2,5,6: closest-to-ambulance (kortste safe travel time)
            if heuristic_id in (2, 5, 6):
                key = (d,)
            else:
                key = (self._severity_rank(getattr(c, "current_triage", "green")), d)


            if (best is None) or (key < best_key):
                best = c
                best_tt = d
                best_key = key

        if best is None:
            return None, None, None

        chosen_path = shortest_path_safe(m, a.node, best.x, best.y, m.t)
        if chosen_path is None:
            return None, None, None

        return best, chosen_path, best_tt

    # -------------------------
    # Override: heli target keuze met feasibility filter
    # -------------------------

    def _heli_best_target(self, h, candidates, m, heuristic_id, claimed):
        if not candidates:
            return None

        # Map heuristics for heli behavior
        # Heuristic 6 must be identical to 5 for target selection (severity then MAX amb_time_from_base)
        if heuristic_id in (3, 5, 6):
            hid = 1
        elif heuristic_id in (4,):
            hid = 2
        else:
            hid = heuristic_id  # 1 or 2

        cand = [
            c for c in candidates
            if (not getattr(c, "assigned", False)) and (id(c) not in claimed)
        ]
        if not cand:
            return None

        cand = self._prefilter_candidates_for_heli(cand, m)

        reachable = [c for c in cand if self._heli_reachable_cached(h, c, m)]
        if not reachable:
            return None

        ttick = int(m.t)
        best = None
        best_key = None

        for c in reachable:
            if not self._heli_can_complete(h, c, m, ttick):
                continue

            amb_t = self._amb_reach_time_from_base(c, m)
            if amb_t is None:
                continue

            if hid == 2:
                key = (-amb_t,)
            else:
                sev = self._severity_rank(getattr(c, "current_triage", "green"))
                key = (sev, -amb_t)

            if (best is None) or (key < best_key):
                best = c
                best_key = key

        return best


    def step(self):
        m = self.model
        self._proactive_avoid_future_no_go(m)
        super().step()

class Ambulance(ap.Agent):
    """
    Ambulance agent (network + offroad) with:
    - safe routing (avoid no-go)
    - capacity/cargo + return-to-base
    - pickup delay (AMBULANCE_PICKUP_DELAY_MIN)
    """

    def setup(self):
        self.x = self.model.base_x
        self.y = self.model.base_y
        self.node = self.model.base_node
        self.path = []
        self.busy = False
        self.needs_route = False

        # Offroad logic
        self.offroad = False
        self.offroad_mode = None          # "to_target" or "to_network"
        self.reentry_node = None
        self.target = None

        # Network traversal state
        self.current_edge = None
        self.edge_time_left = 0.0

        # Movement logging
        self.prev_x = self.x
        self.prev_y = self.y
        self.speed = 0.0

        # Capacity / cargo
        self.capacity = self.model.AMBULANCE_CAPACITY
        self.cargo = []  # list of casualty agents

        # Return-to-base state (via network)
        self.returning = False
        self.return_path = []
        self.return_edge = None
        self.return_edge_time_left = 0.0

        # Pickup delay (+1 min)
        self.picking_up = False
        self.pickup_timer = 0.0
        self.pickup_target = None

        # Track whether we ever had a reachable (safe) network path to current target
        self.had_network_path = False

        self._was_in_no_go = False
        self.disabled = False

        self.return_after_next_pickup = False


    # -------------------------
    # Helpers
    # -------------------------

    def _soft_retreat_from_no_go(self, reason="entered_no_go_retreat"):
        """
        SOFT doctrine:
        - geen disable/failure
        - abort missie + pickup
        - move off-road to the nearest safe node (reentry_node)
        """
        m = self.model

        # Optionally log the incident without increasing the failure count.
        if hasattr(m, "platform_failure_events"):
            m.platform_failure_events.append({
                "t": float(m.t),
                "platform": "ambulance",
                "agent_id": int(getattr(self, "id", id(self))),
                "x": float(self.x),
                "y": float(self.y),
                "reason": str(reason),
            })

        # free target
        if self.target is not None:
            self.target.assigned = False

        # reset mission state
        self.busy = False
        self.target = None
        self.path = []
        self.current_edge = None
        self.edge_time_left = 0.0
        self.had_network_path = False

        # Stop returning because the platform must first move to safety.
        self.returning = False
        self.return_path = []
        self.return_edge = None
        self.return_edge_time_left = 0.0

        # stop pickup
        self.picking_up = False
        self.pickup_timer = 0.0
        self.pickup_target = None

        # force offroad naar nearest safe node
        self.offroad = True
        self.offroad_mode = "to_network"
        self.reentry_node = self._nearest_safe_node_now()

    def _disable_platform(self, reason="entered_no_go_tile"):
        m = self.model

        # metrics (1x)
        m.record_platform_failure(
            platform_type="ambulance",
            agent_id=getattr(self, "id", id(self)),
            t=m.t,
            x=self.x,
            y=self.y,
            reason=reason
        )

        # free casualty if assigned/targeted
        if self.target is not None:
            self.target.assigned = False

        # drop everything
        self.disabled = True
        self.busy = False
        self.returning = False
        self.offroad = False
        self.offroad_mode = None
        self.reentry_node = None

        self.target = None
        self.path = []
        self.current_edge = None
        self.edge_time_left = 0.0

        self.return_path = []
        self.return_edge = None
        self.return_edge_time_left = 0.0

        self.picking_up = False
        self.pickup_timer = 0.0
        self.pickup_target = None

        # decide what happens to cargo:
        # option A: cargo is lost (most "failure-ish")
        # self.cargo = []

        # option B: keep cargo but vehicle can't move anymore (so they'll never arrive)
        # (do nothing)

    def _abort_current_task(self, unassign=True, reason="no_go_block"):
        """Abort current rescue task and reset movement state."""
        m = self.model

        # log mission abort only if there was an active target
        if self.target is not None:
            m.record_mission_abort(
                kind=reason,
                platform_type="ambulance",
                agent_id=getattr(self, "id", id(self)),
                casualty=self.target,
                t=m.t,
                x=float(self.x),
                y=float(self.y),
            )

        if unassign and (self.target is not None):
            self.target.assigned = False

        self.target = None
        self.busy = False
        self.path = []
        self.current_edge = None
        self.edge_time_left = 0.0
        self.had_network_path = False

        # reset offroad navigation back to network
        self.offroad = True
        self.offroad_mode = "to_network"
        self.reentry_node = self._nearest_safe_node_now()

        # stop pickup if it was ongoing
        self.picking_up = False
        self.pickup_timer = 0.0
        self.pickup_target = None


    def _start_return_to_base(self):
        m = self.model
        self.returning = True
        forbidden = m.get_forbidden_nodes(m.t)
        H = m.get_safe_graph(int(m.t))
        # bereikbaarheidstest in beide richtingen
        try:
            can_src_to_base = nx.has_path(H, self.node, m.base_node)
        except Exception as e:
            print("has_path src->base error", e)
            can_src_to_base = None

        try:
            can_base_to_src = nx.has_path(H, m.base_node, self.node)
        except Exception as e:
            print("has_path base->src error", e)
            can_base_to_src = None

        if H.is_directed():
            base_in = H.in_degree(m.base_node)
            base_out = H.out_degree(m.base_node)
        else:
            base_in = base_out = H.degree(m.base_node)




        # safe return path (avoid no-go)
        self.return_path = shortest_path_to_node_safe(m, self.node, m.base_node, m.t)

        self.return_edge = None
        self.return_edge_time_left = 0.0

        # Stop current rescue task
        self.busy = False
        self.target = None
        self.path = []
        self.current_edge = None
        self.edge_time_left = 0.0
        self.had_network_path = False

        # Stop pickup
        self.picking_up = False
        self.pickup_timer = 0.0
        self.pickup_target = None

        # reset offroad
        self.offroad = False
        self.offroad_mode = None
        self.reentry_node = None

    def _deliver_if_at_base(self):
        m = self.model
        if euclid((self.x, self.y), (m.base_x, m.base_y)) < 2:
            for c in self.cargo:
                m.base.admit(c, m.t)
            m.delivered_total += len(self.cargo)
            self.cargo = []

            self.returning = False
            self.return_path = []
            self.return_edge = None
            self.return_edge_time_left = 0.0
            return True
        return False

    def _finalize_pickup(self):
        """Finish pickup after the extra pickup-time."""
        m = self.model
        t = self.pickup_target

        # --- helper: reset pickup state ---
        def reset_pickup_state():
            self.picking_up = False
            self.pickup_timer = 0.0
            self.pickup_target = None

        # --- helper: reset mission & motion state + go back to network ---
        def abort_and_go_back(unassign=True):
            if unassign and (self.target is not None):
                self.target.assigned = False

            # after abort: go back to network
            self.offroad = True
            self.offroad_mode = "to_network"
            self.reentry_node = self._nearest_safe_node_now()

            # stop task
            self.busy = False
            self.target = None
            self.path = []
            self.current_edge = None
            self.edge_time_left = 0.0
            self.had_network_path = False

        # -------------------------
        # 1) invalid target / already taken
        # -------------------------
        if (t is None) or getattr(t, "evacuated", False):
            # Optionally log a generic abort event if desired.
            # if t is not None:
            #     m.record_mission_abort("no_go_block", "ambulance", getattr(self, "id", id(self)), t, m.t)

            reset_pickup_state()
            abort_and_go_back(unassign=False)
            return

        # -------------------------
        # 2) casualty became black BEFORE pickup completed -> abort
        # -------------------------
        if getattr(t, "current_triage", None) == "black":
            m.record_mission_abort(
                kind="target_became_black_before_pickup",
                platform_type="ambulance",
                agent_id=getattr(self, "id", id(self)),
                casualty=t,
                t=m.t,
                x=float(self.x),
                y=float(self.y),
            )
            reset_pickup_state()
            abort_and_go_back(unassign=True)
            return

        # -------------------------
        # 3) tile became no-go at pickup completion -> abort
        # -------------------------
        if is_no_go_at_time(m, t.x, t.y, m.t):
            m.record_mission_abort(
                kind="target_became_no_go",
                platform_type="ambulance",
                agent_id=getattr(self, "id", id(self)),
                casualty=t,
                t=m.t,
                x=float(self.x),
                y=float(self.y),
            )
            reset_pickup_state()
            abort_and_go_back(unassign=True)
            return

        # -------------------------
        # 4) success: pick up
        # -------------------------
        t.evacuated = True
        t.state = "transport"
        t.pickup_time = m.t
        t.picked_by = "ambulance"  # metrics

        self.cargo.append(t)

        # clear pickup state
        reset_pickup_state()

        # If we were doing a "second pickup" (cargo already had 1),
        # force return after we re-enter the network.
        if getattr(self, "return_after_next_pickup", False) or (len(self.cargo) >= self.capacity):
            self.returning = True
            self.return_after_next_pickup = False


        # after pickup: go back to network
        self.offroad = True
        self.offroad_mode = "to_network"
        self.reentry_node = self._nearest_safe_node_now()

        # task done
        self.busy = False
        self.target = None
        self.path = []
        self.current_edge = None
        self.edge_time_left = 0.0
        self.had_network_path = False

    def _nearest_safe_node_now(self):
        """Nearest graph node that is NOT in no-go at current time."""
        m = self.model
        forbidden = build_forbidden_nodes_at_time(m, m.t)
        n = nearest_node_safe(m, self.x, self.y, forbidden)
        # fallback: if everything forbidden (rare), use plain nearest
        return n if n is not None else nearest_node(self.x, self.y)

    # -------------------------
    # Main step
    # -------------------------
    def step(self):
        m = self.model

        # -------------------------
        # If already disabled: do nothing forever
        # -------------------------
        if self.disabled:
            self.speed = 0.0
            self.prev_x, self.prev_y = self.x, self.y
            return

        # -------------------------
        # Platform failure: entered no-go (1x)
        # -------------------------
        now_in_no_go = is_no_go_at_time(m, self.x, self.y, m.t)
        if now_in_no_go and (not self._was_in_no_go):
            doctrine = str(getattr(m, "PLATFORM_DOCTRINE", "HARD")).upper()

            if doctrine == "HARD":
                self._disable_platform(reason="entered_no_go_tile")
                self._was_in_no_go = True
                self.speed = 0.0
                self.prev_x, self.prev_y = self.x, self.y
                return

            # SOFT doctrine
            self._soft_retreat_from_no_go(reason="entered_no_go_retreat")
            self._was_in_no_go = True
            # niet returnen: je offroad logic onderin laat hem meteen richting safe node bewegen
        else:
            self._was_in_no_go = now_in_no_go

        # -------------------------
        # Plan route if needed (only when on-network)
        # -------------------------
        if self.busy and (not self.offroad) and self.needs_route:
            t = self.target
            if t is None or t.evacuated or is_no_go_at_time(m, t.x, t.y, m.t):
                self._abort_current_task(unassign=True)
            else:
                p = shortest_path_safe(m, self.node, t.x, t.y, m.t)
                if p is None:
                    self._abort_current_task(unassign=True)
                else:
                    self.path = p
                    self.needs_route = False

        # -------------------------
        # If currently in a no-go tile: emergency exit to safe network
        # -------------------------
        if is_no_go_at_time(m, self.x, self.y, m.t):
            if self.target is not None:
                self.target.assigned = False
            self.target = None
            self.busy = False
            self.path = []
            self.current_edge = None
            self.edge_time_left = 0.0
            self.had_network_path = False

            # stop pickup
            self.picking_up = False
            self.pickup_timer = 0.0
            self.pickup_target = None

            # force offroad to nearest SAFE node
            self.offroad = True
            self.offroad_mode = "to_network"
            self.reentry_node = self._nearest_safe_node_now()

            # cancel return until safe again
            self.returning = False
            self.return_path = []
            self.return_edge = None
            self.return_edge_time_left = 0.0

        # -------------------------
        # NO-GO dynamic failsafe (enroute)
        # -------------------------
        if self.busy and (self.target is not None):
            if is_no_go_at_time(m, self.target.x, self.target.y, m.t):
                self._abort_current_task(unassign=True, reason="target_became_no_go")

        # -------------------------
        # Dispatch fallback (normally Coordinator does this)
        # -------------------------
        if (not hasattr(m, "coordinator")) or (m.coordinator is None):
            if (not self.busy) and (not self.offroad) and (not self.returning) and (not self.picking_up):
                if len(self.cargo) >= self.capacity:
                    self._start_return_to_base()
                else:
                    candidates = [
                        c for c in m.casualties
                        if (not c.evacuated)
                        and (not c.assigned)
                        and (not is_no_go_at_time(m, c.x, c.y, m.t))
                    ]
                    if candidates:
                        t = min(candidates, key=lambda c: euclid((self.x, self.y), (c.x, c.y)))
                        t.assigned = True
                        self.target = t

                        p = shortest_path_safe(m, self.node, t.x, t.y, m.t)
                        if p is None:
                            t.assigned = False
                            self.target = None
                            self.busy = False
                            self.path = []
                            self.had_network_path = False
                        else:
                            self.path = p
                            self.had_network_path = True
                            self.busy = True
                            self.current_edge = None
                            self.edge_time_left = 0.0

                            if len(self.path) == 0:
                                self.offroad = True
                                self.offroad_mode = "to_target"
                    else:
                        if len(self.cargo) > 0:
                            self._start_return_to_base()

        # -------------------------
        # Tick budget (minutes)
        # -------------------------
        time_left = float(m.TIMESTEP_DURATION)

        # -------------------------
        # Pickup delay ticking (ambulance stands still)
        # -------------------------
        if self.picking_up:
            self.pickup_timer -= m.TIMESTEP_DURATION
            if self.pickup_timer <= 0:
                self._finalize_pickup()

            dx = self.x - self.prev_x
            dy = self.y - self.prev_y
            self.speed = math.hypot(dx, dy) / m.TIMESTEP_DURATION
            self.prev_x, self.prev_y = self.x, self.y
            return

        # -------------------------
        # Return-to-base over network (NO early return!)
        # -------------------------
        while self.returning and (not self.offroad) and time_left > 0:
            if self._deliver_if_at_base():
                break

            if self.return_edge is None:
                # distinguish no path (None) vs done ([])
                if self.return_path is None:
                    # no safe network path -> go offroad to safe node
                    self.offroad = True
                    self.offroad_mode = "to_network"
                    self.reentry_node = self._nearest_safe_node_now()
                    break

                if len(self.return_path) == 0:
                    if not self._deliver_if_at_base():
                        self.returning = False
                    break

                self.return_edge = self.return_path.pop(0)
                tt = get_edge_travel_time(self.node, self.return_edge)
                if tt is None:
                    # inconsistent path -> replan return
                    self._start_return_to_base()
                    break
                self.return_edge_time_left = float(tt)

            # advance along current return edge
            if time_left >= self.return_edge_time_left:
                time_left -= self.return_edge_time_left
                self.node = self.return_edge

                forbidden = build_forbidden_nodes_at_time(m, m.t)
                if self.node in forbidden:
                    safe = nearest_node_safe(m, self.x, self.y, forbidden)
                    if safe is not None:
                        self.node = safe
                        self.x = G.nodes[self.node]["x"]
                        self.y = G.nodes[self.node]["y"]

                    # clear movement state and go offroad-to-network
                    self.current_edge = None
                    self.edge_time_left = 0.0
                    self.return_edge = None
                    self.return_edge_time_left = 0.0
                    self.offroad = True
                    self.offroad_mode = "to_network"
                    self.reentry_node = self._nearest_safe_node_now()
                    break

                self.x = G.nodes[self.node]["x"]
                self.y = G.nodes[self.node]["y"]
                self.return_edge = None
                self.return_edge_time_left = 0.0
            else:
                u, v = self.node, self.return_edge
                frac = time_left / self.return_edge_time_left
                self.x += frac * (G.nodes[v]["x"] - G.nodes[u]["x"])
                self.y += frac * (G.nodes[v]["y"] - G.nodes[u]["y"])
                self.return_edge_time_left -= time_left
                time_left = 0.0

        # -------------------------
        # Move to target over network
        # -------------------------
        while self.busy and (not self.offroad) and time_left > 0:
            if self.current_edge is None:
                if self.path == []:
                    self.offroad = True
                    self.offroad_mode = "to_target"
                    break

                self.current_edge = self.path.pop(0)
                tt = get_edge_travel_time(self.node, self.current_edge)
                if tt is None:
                    # path inconsistent -> replan if target exists
                    if self.target is not None:
                        p = shortest_path_safe(m, self.node, self.target.x, self.target.y, m.t)
                        if p is None:
                            self._abort_current_task(unassign=True)
                            break
                        self.path = p
                        self.current_edge = None
                        self.edge_time_left = 0.0
                        continue
                    else:
                        self.busy = False
                        self.path = []
                        self.current_edge = None
                        self.edge_time_left = 0.0
                        break

                self.edge_time_left = float(tt)

            # advance along current edge
            if time_left >= self.edge_time_left:
                time_left -= self.edge_time_left
                self.node = self.current_edge

                forbidden = build_forbidden_nodes_at_time(m, m.t)
                if self.node in forbidden:
                    safe = nearest_node_safe(m, self.x, self.y, forbidden)
                    if safe is not None:
                        self.node = safe
                        self.x = G.nodes[self.node]["x"]
                        self.y = G.nodes[self.node]["y"]

                    # clear movement state and go offroad-to-network
                    self.current_edge = None
                    self.edge_time_left = 0.0
                    self.return_edge = None
                    self.return_edge_time_left = 0.0
                    self.offroad = True
                    self.offroad_mode = "to_network"
                    self.reentry_node = self._nearest_safe_node_now()
                    break

                self.x = G.nodes[self.node]["x"]
                self.y = G.nodes[self.node]["y"]
                self.current_edge = None
                self.edge_time_left = 0.0
            else:
                u, v = self.node, self.current_edge
                frac = time_left / self.edge_time_left
                self.x += frac * (G.nodes[v]["x"] - G.nodes[u]["x"])
                self.y += frac * (G.nodes[v]["y"] - G.nodes[u]["y"])
                self.edge_time_left -= time_left
                time_left = 0.0

        # -------------------------
        # Offroad behavior (use remaining time_left, not full timestep)
        # -------------------------
        if self.offroad and time_left > 0:
            if self.offroad_mode == "to_target":
                t = self.target
                if t is None:
                    self.offroad = True
                    self.offroad_mode = "to_network"
                    self.reentry_node = self._nearest_safe_node_now()
                else:
                    if is_no_go_at_time(m, t.x, t.y, m.t):
                        self._abort_current_task(unassign=True)
                    else:
                        dx, dy = t.x - self.x, t.y - self.y
                        d = math.hypot(dx, dy)
                        if d < 2:
                            self.picking_up = True
                            self.pickup_timer = float(m.AMBULANCE_PICKUP_DELAY_MIN)
                            self.pickup_target = t
                        else:
                            step = min(m.OFFROAD_SPEED * time_left, d)
                            if d > 0:
                                nx_ = self.x + step * dx / d
                                ny_ = self.y + step * dy / d

                                # prevent entering no-go while offroad
                                if is_no_go_at_time(m, nx_, ny_, m.t):
                                    forbidden = build_forbidden_nodes_at_time(m, m.t)
                                    new_tgt = nearest_approach_node_safe(
                                        m, t.x, t.y, forbidden, m.t, k=50, step_m=25.0
                                    )
                                    if new_tgt is None:
                                        self._abort_current_task(unassign=True)
                                    else:
                                        p = shortest_path_safe(
                                            m, self.node,
                                            G.nodes[new_tgt]["x"], G.nodes[new_tgt]["y"],
                                            m.t
                                        )
                                        if p is None:
                                            self._abort_current_task(unassign=True)
                                        else:
                                            self.path = p
                                            self.busy = True
                                            self.offroad = False
                                            self.offroad_mode = None
                                            self.reentry_node = None
                                            self.current_edge = None
                                            self.edge_time_left = 0.0
                                else:
                                    self.x, self.y = nx_, ny_

            elif self.offroad_mode == "to_network":
                n = self.reentry_node
                nx_, ny_ = G.nodes[n]["x"], G.nodes[n]["y"]
                dx, dy = nx_ - self.x, ny_ - self.y
                d = math.hypot(dx, dy)
                if d < 2:
                    self.x, self.y = nx_, ny_
                    self.node = n

                    self.offroad = False
                    self.offroad_mode = None
                    self.reentry_node = None

                    if self.returning:
                        self._start_return_to_base()
                    elif len(self.cargo) >= self.capacity:
                        self._start_return_to_base()
                else:
                    step = min(m.OFFROAD_SPEED * time_left, d)
                    if d > 0:
                        nx2 = self.x + step * dx / d
                        ny2 = self.y + step * dy / d

                        if is_no_go_at_time(m, nx2, ny2, m.t):
                            self.reentry_node = self._nearest_safe_node_now()
                        else:
                            self.x, self.y = nx2, ny2

        # -------------------------
        # Speed logging (ONLY once, at end)
        # -------------------------
        dxm = self.x - self.prev_x
        dym = self.y - self.prev_y
        self.speed = math.hypot(dxm, dym) / m.TIMESTEP_DURATION
        self.prev_x, self.prev_y = self.x, self.y

class Helicopter(ap.Agent):
    def setup(self):
        m = self.model

        self._was_in_no_go = False
        self.disabled = False

        # Position & motion
        self.x = m.base_x
        self.y = m.base_y
        self.prev_x = self.x
        self.prev_y = self.y
        self.speed = 0.0

        # Tasking
        self.busy = False
        self.target = None
        self.returning = False

        # Cargo
        self.capacity = m.HELICOPTER_CAPACITY
        self.cargo = []

        # Pickup delay state
        self.picking_up = False
        self.pickup_timer = 0.0
        self.pickup_target = None

        self.retreating = False
        self.retreat_x = None
        self.retreat_y = None

    # -------------------------
    # Helpers
    # -------------------------
    def _disable_platform(self, reason="entered_no_go_tile"):
        m = self.model

        m.record_platform_failure(
            platform_type="helicopter",
            agent_id=getattr(self, "id", id(self)),
            t=m.t,
            x=self.x,
            y=self.y,
            reason=reason
        )

        if self.target is not None:
            self.target.assigned = False

        self.disabled = True
        self.busy = False
        self.returning = False
        self.target = None

        self.picking_up = False
        self.pickup_timer = 0.0
        self.pickup_target = None

        # Cargo policy follows the same rule as the ambulance.
        # self.cargo = []

    def _log_speed(self):
        m = self.model
        dxm = self.x - self.prev_x
        dym = self.y - self.prev_y
        self.speed = math.hypot(dxm, dym) / m.TIMESTEP_DURATION
        self.prev_x, self.prev_y = self.x, self.y

    def _abort_current_mission(self, reason="no_go_block"):
        m = self.model
        if self.target is not None:
            m.record_mission_abort(
                kind=reason,
                platform_type="helicopter",
                agent_id=getattr(self, "id", id(self)),
                casualty=self.target,
                t=m.t,
                x=float(self.x), y=float(self.y),
            )
            self.target.assigned = False

        self.busy = False
        self.target = None

        self.picking_up = False
        self.pickup_timer = 0.0
        self.pickup_target = None

    def _deliver_if_at_base(self):
        """If at base: unload all cargo."""
        m = self.model
        if euclid((self.x, self.y), (m.base_x, m.base_y)) < 2:
            for c in self.cargo:
                m.base.admit(c, m.t)
            m.delivered_total += len(self.cargo)
            self.cargo = []
            self.returning = False
            return True
        return False

    def _finalize_pickup(self):
        """
        Finish pickup after delay.
        Variant A: black is NOT allowed -> abort + metric.
        """
        m = self.model
        t = self.pickup_target

        def reset_pickup_state():
            self.picking_up = False
            self.pickup_timer = 0.0
            self.pickup_target = None

        def abort_mission(unassign=True, reason="no_go_block"):
            # log if we had a target
            if self.target is not None:
                m.record_mission_abort(
                    kind=reason,
                    platform_type="helicopter",
                    agent_id=getattr(self, "id", id(self)),
                    casualty=self.target,
                    t=m.t,
                    x=float(self.x),
                    y=float(self.y),
                )
                if unassign:
                    self.target.assigned = False

            self.busy = False
            self.target = None

        # invalid/already evacuated
        if (t is None) or getattr(t, "evacuated", False):
            reset_pickup_state()
            abort_mission(unassign=False, reason="no_go_block")
            return

        # black before pickup done -> abort
        if getattr(t, "current_triage", None) == "black":
            reset_pickup_state()
            abort_mission(unassign=True, reason="target_became_black_before_pickup")
            return

        # tile no-go -> abort
        if is_no_go_at_time(m, t.x, t.y, m.t):
            reset_pickup_state()
            abort_mission(unassign=True, reason="target_became_no_go")
            return

        # success
        t.evacuated = True
        t.state = "transport"
        t.pickup_time = m.t
        t.picked_by = "helicopter"

        self.cargo.append(t)

        reset_pickup_state()
        self.busy = False
        self.target = None

        if len(self.cargo) >= self.capacity:
            self.returning = True

    def _soft_retreat_from_no_go(self, reason="entered_no_go_retreat"):
        m = self.model

        # (optioneel) log incident zonder failure-count
        if hasattr(m, "platform_failure_events"):
            m.platform_failure_events.append({
                "t": float(m.t),
                "platform": "helicopter",
                "agent_id": int(getattr(self, "id", id(self))),
                "x": float(self.x),
                "y": float(self.y),
                "reason": str(reason),
            })

        # abort mission + pickup
        self._abort_current_mission()

        # Choose the nearest safe node as a waypoint.
        tt = int(m.t)
        safe_n = nearest_node_safe_at_time(m, self.x, self.y, tt)  # jij gebruikt dit al elders
        if safe_n is None:
            # fallback: base
            self.retreat_x, self.retreat_y = float(m.base_x), float(m.base_y)
        else:
            self.retreat_x = float(G.nodes[safe_n]["x"])
            self.retreat_y = float(G.nodes[safe_n]["y"])

        self.retreating = True
        self.returning = False   # we willen naar nearest safe, niet per se base
        self.busy = False
    # -------------------------
    # Main step
    # -------------------------
    def step(self):
        m = self.model

        if self.disabled:
            self.speed = 0.0
            self.prev_x, self.prev_y = self.x, self.y
            return

        now_in_no_go = is_no_go_at_time(m, self.x, self.y, m.t)
        if now_in_no_go and (not self._was_in_no_go):
            doctrine = str(getattr(m, "PLATFORM_DOCTRINE", "HARD")).upper()

            if doctrine == "HARD":
                self._disable_platform(reason="entered_no_go_tile")
                self._was_in_no_go = True
                self.speed = 0.0
                self.prev_x, self.prev_y = self.x, self.y
                return

            # SOFT doctrine
            self._soft_retreat_from_no_go(reason="entered_no_go_retreat")
            self._was_in_no_go = True
        else:
            self._was_in_no_go = now_in_no_go


        self._was_in_no_go = now_in_no_go




        # 1) If heli is in a no-go zone, abort and retreat to base
        if is_no_go_at_time(m, self.x, self.y, m.t):
            self._abort_current_mission(reason="no_go_block")
            self.returning = True


        # 2) Fallback dispatch if no coordinator exists
        if (not hasattr(m, "coordinator")) or (m.coordinator is None):
            # If full: return
            if (not self.returning) and (len(self.cargo) >= self.capacity):
                self.returning = True
                self.busy = False
                self.target = None

            # If idle and has capacity: pick a target
            if (not self.busy) and (not self.returning) and (len(self.cargo) < self.capacity):
                candidates = [
                    c for c in m.casualties
                    if (not getattr(c, "evacuated", False))
                    and (not getattr(c, "assigned", False))
                    and (not is_no_go_at_time(m, c.x, c.y, m.t))
                ]
                if candidates:
                    # your logic: farthest from current heli position
                    t = max(candidates, key=lambda c: euclid((self.x, self.y), (c.x, c.y)))
                    t.assigned = True
                    self.target = t
                    self.busy = True
                else:
                    # No targets. If carrying anyone, return; else idle at current position.
                    if len(self.cargo) > 0:
                        self.returning = True

        # 3) Pickup delay ticking (heli stands still during pickup)
        if self.picking_up:
            self.pickup_timer -= m.TIMESTEP_DURATION
            if self.pickup_timer <= 0:
                self._finalize_pickup()
            self._log_speed()
            return
        
        # --- SOFT retreat behavior ---
        if getattr(self, "retreating", False):
            tx, ty = float(self.retreat_x), float(self.retreat_y)
            dx, dy = tx - self.x, ty - self.y
            d = math.hypot(dx, dy)

            if d < 2:
                # Arrived at the safe waypoint.
                self.retreating = False
                self.retreat_x = None
                self.retreat_y = None
                # After this, the coordinator can dispatch the platform again.
            else:
                step = min(m.HELI_SPEED * m.TIMESTEP_DURATION, d)
                if d > 0:
                    self.x += step * dx / d
                    self.y += step * dy / d

            self._log_speed()
            return


        # 4) Return-to-base behavior
        if self.returning:
            if not self._deliver_if_at_base():
                dx, dy = m.base_x - self.x, m.base_y - self.y
                d = math.hypot(dx, dy)
                step = min(m.HELI_SPEED * m.TIMESTEP_DURATION, d)
                if d > 0:
                    self.x += step * dx / d
                    self.y += step * dy / d
            self._log_speed()
            return

        # 5) Fly to target & initiate pickup when close
        if self.busy:
            t = self.target

            # Abort if invalid / already evacuated / turned no-go
            if (t is None) or getattr(t, "evacuated", False):
                self._abort_current_mission(reason="no_go_block")
            elif is_no_go_at_time(m, t.x, t.y, m.t):
                self._abort_current_mission(reason="target_became_no_go")

                if t is not None:
                    t.assigned = False
                self.busy = False
                self.target = None
            else:
                dx, dy = t.x - self.x, t.y - self.y
                d = math.hypot(dx, dy)

                if d < 2:
                    # start pickup delay
                    self.picking_up = True
                    self.pickup_timer = float(m.HELICOPTER_PICKUP_DELAY_MIN)
                    self.pickup_target = t
                else:
                    # move towards target
                    step = min(m.HELI_SPEED * m.TIMESTEP_DURATION, d)
                    if d > 0:
                        self.x += step * dx / d
                        self.y += step * dy / d

        # 6) Speed logging
        self._log_speed()

class CasevacModel(ap.Model):

    def setup(self):
        self.NUM_AMBULANCES = int(globals().get("NUM_AMBULANCES", 8))
        self.NUM_HELICOPTERS = int(globals().get("NUM_HELICOPTERS", 1))
        self.NUM_CASUALTIES = 0

        self.TIME_STEPS = TIME_STEPS
        self.TIMESTEP_DURATION = 1.0  # minutes per tick
        self.OFFROAD_SPEED = 5 * 1000 / 60
        self.HELI_SPEED = 180 * 1000 / 60

        # Pickup delays in additional minutes.
        self.AMBULANCE_PICKUP_DELAY_MIN = 1.0
        self.HELICOPTER_PICKUP_DELAY_MIN = 3.0

        # Capacities
        self.AMBULANCE_CAPACITY = 2
        self.HELICOPTER_CAPACITY = 4

        # Base treatment parameters, mapping older names to current model fields.
        self.N_BEDS = 10
        # scenario-controlled base policy (from globals)
        self.BASE_POLICY = str(globals().get("BASE_POLICY", Base_Policy)).upper()  # "SEVERITY" / "FIFO"

        # scenario-controlled dispatch heuristic (from globals)
        self.DISPATCH_HEURISTIC = int(globals().get("DISPATCH_HEURISTIC", DISPATCH_HEURISTIC))  # 1 / 2

        self.PLATFORM_DOCTRINE = str(globals().get("PLATFORM_DOCTRINE", "HARD")).upper()

        xs = [G.nodes[n]["x"] for n in nodes]
        ys = [G.nodes[n]["y"] for n in nodes]
        self.xmin, self.xmax = min(xs), max(xs)
        self.ymin, self.ymax = min(ys), max(ys)

        # -------------------------
        # Dynamic tiles accessible in AgentPy model
        # -------------------------
        self.tile_snapshots = tile_snapshots  # uses the global snapshots you already build

        # seed-level flip metrics uit environment loop
        self.tile_flip_history = globals().get("tile_flip_history", [])
        self.tile_flip_totals  = globals().get("tile_flip_totals", [])
        self.tile_share_history = globals().get("tile_share_history", [])

        self.access_history = {
            "t": [],
            "forbidden_share_nodes": [],
            "safe_lcc_share": [],
            "reachable_from_base_share": [],
        }

        # -------------------------
        # Fight situation time series (F_t, S_t) per seed/run
        # -------------------------
        # Prefer self.p.seed when using an AgentPy seed parameter.
        # Fall back to the global seed when that is the configured approach.

        self.F_ts = globals().get("F_ts", None)
        self.S_ts = globals().get("S_ts", None)


        # -------------------------
        # Unreachable / blocked casualty metrics
        # -------------------------
        self.unreachable = {
            # unique counts (we mark on casualty to avoid double-counting)
            "spawned_in_no_go_unique": 0,
            "filtered_no_go_current_unique": 0,
            "amb_no_approach_unique": 0,
            "amb_no_path_unique": 0,
            "heli_line_blocked_unique": 0,

            # optional: frequency counts (per tick / per evaluation)
            "filtered_no_go_current_freq": 0,
            "amb_no_approach_freq": 0,
            "amb_no_path_freq": 0,
            "heli_line_blocked_freq": 0,
        }


        # Precompute node -> grid indices (for forbidden nodes lookup)
        self._node_grid_ix = {}
        self._node_grid_iy = {}
        for n in nodes:
            xw = float(G.nodes[n]["x"])
            yw = float(G.nodes[n]["y"])
            ix, iy = world_to_grid(xw, yw, x_min, y_min, Dx, Dy)

            if ix < 0 or iy < 0:
                self._node_grid_ix[n] = None
                self._node_grid_iy[n] = None
            else:
                self._node_grid_ix[n] = int(ix)
                self._node_grid_iy[n] = int(iy)

        # ------------------------------------------------------------
        # Base must start on a GREEN tile (tile class 0) at t=0
        # ------------------------------------------------------------
        t0 = 0
        green_nodes = [
            n for n in nodes
            if tile_class_at_time(self, float(G.nodes[n]["x"]), float(G.nodes[n]["y"]), t0) == 0
        ]

        if not green_nodes:
            raise RuntimeError("No GREEN nodes available for base at t=0.")

        self.base_node = random.choice(green_nodes)
        self.base_x = float(G.nodes[self.base_node]["x"])
        self.base_y = float(G.nodes[self.base_node]["y"])


        # Base agent (nu weer echt behandelen)
        self.base = Base(self)

        # Coordinator keuxze
        self.USE_CLAIRVOYANT_COORDINATOR = bool(globals().get("USE_CLAIRVOYANT_COORDINATOR", False))

        if self.USE_CLAIRVOYANT_COORDINATOR:
            self.coordinator = ClairvoyantCoordinator(self)
        else:
            self.coordinator = Coordinator(self)



        self.casualties = ap.AgentList(self, 0, Casualty)
        self.ambulances = ap.AgentList(self, self.NUM_AMBULANCES, Ambulance)
        self.helicopters = ap.AgentList(self, self.NUM_HELICOPTERS, Helicopter)

        self.delivered_total = 0
        self.history = []

        self._forbidden_cache_t = None
        self._forbidden_cache = None
        self._safe_graph_cache_t = None
        self._safe_graph_cache = None

        # metrics collector (created once)
        self.metrics = MetricsCollector(self)

        self._node_ids = np.array(list(nodes))
        coords = np.array([(G.nodes[n]["x"], G.nodes[n]["y"]) for n in self._node_ids], dtype=float)
        self._node_xy = coords
        self._kdtree = cKDTree(coords)
        self.BASE_DIJKSTRA_CUTOFF = 55.0

        self.PREFILTER_M = 9        # ambulance top-M
        self.HELI_PREFILTER_M = 6     # heli top-M (maak klein!)
        self.HELI_STEP_M = 120.0       # minder samples in segment_crosses_no_go
        self.HELI_SCORE_M = 8

        # --- Platform failure metrics ---
        self.platform_failures = {
            "ambulance": 0,
            "helicopter": 0,
        }
        self.platform_failure_events = []  # optional: detailed log
        self.base_arrivals = []   # list of dicts
        self._cid_seq = 0

        # --- Mission abort metrics ---
        self.mission_aborts = {
            "black_on_arrival": 0,                 # casualty arrives black at base
            "no_go_block": 0,                      # mission cannot continue due to no-go dynamics
            "target_became_no_go": 0,              # target tile became no-go while enroute
            "target_became_black_before_pickup": 0,# casualty turned black before pickup completed
        }
        self.mission_abort_events = []  # optional detailed log

    def record_platform_failure(self, platform_type, agent_id, t, x, y, reason="entered_no_go"):
        self.platform_failures[platform_type] = self.platform_failures.get(platform_type, 0) + 1
        self.platform_failure_events.append({
            "t": float(t),
            "platform": str(platform_type),
            "agent_id": int(agent_id),
            "x": float(x),
            "y": float(y),
            "reason": str(reason),
        })

    def record_mission_abort(self, kind, platform_type=None, agent_id=None, casualty=None, t=None, **extra):
        # counters
        if kind not in self.mission_aborts:
            self.mission_aborts[kind] = 0
        self.mission_aborts[kind] += 1

        # detailed event log (optional, but super useful)
        cid = None
        if casualty is not None:
            cid = int(getattr(casualty, "cid", -1))

        self.mission_abort_events.append({
            "t": float(self.t if t is None else t),
            "kind": str(kind),
            "platform": None if platform_type is None else str(platform_type),
            "agent_id": None if agent_id is None else int(agent_id),
            "cid": cid,
            **extra
        })

    def get_forbidden_nodes(self, t):
        """Return cached forbidden nodes for this tick."""
        tt = int(t)
        if self._forbidden_cache_t == tt and self._forbidden_cache is not None:
            return self._forbidden_cache

        forbidden = build_forbidden_nodes_at_time_uncached(self, tt)
        self._forbidden_cache_t = tt
        self._forbidden_cache = forbidden
        return forbidden

    def get_safe_graph(self, t):
        """Return cached subgraph_view that excludes forbidden nodes for this tick."""
        tt = int(t)
        if self._safe_graph_cache_t == tt and self._safe_graph_cache is not None:
            return self._safe_graph_cache

        forbidden = self.get_forbidden_nodes(tt)
        # NOTE: subgraph_view is lazy, but creating it still costs; do it once per tick
        H = nx.subgraph_view(G, filter_node=lambda n, f=forbidden: n not in f)
        self._safe_graph_cache_t = tt
        self._safe_graph_cache = H
        return H


    def log_access_metrics(self, t):
        """Log network-level access loss for tick t."""
        tt = int(t)

        forbidden = self.get_forbidden_nodes(tt)
        total_nodes = len(nodes)  # uses global nodes list (consistent with rest of code)

        safe_nodes = total_nodes - len(forbidden)
        forbidden_share = (len(forbidden) / total_nodes) if total_nodes else 0.0

        H = self.get_safe_graph(tt)

        # Largest connected component share (treat as undirected)
        try:
            Hu = H.to_undirected(as_view=True)
            # connected_components expects an undirected graph
            largest = max((len(c) for c in nx.connected_components(Hu)), default=0)
            lcc_share = (largest / safe_nodes) if safe_nodes > 0 else 0.0
        except Exception:
            lcc_share = float("nan")

        # Reachable from base share
        try:
            Hu = H.to_undirected(as_view=True)
            if self.base_node in Hu:
                reachable = len(nx.node_connected_component(Hu, self.base_node))
                reachable_share = (reachable / safe_nodes) if safe_nodes > 0 else 0.0
            else:
                reachable_share = 0.0
        except Exception:
            reachable_share = float("nan")

        self.access_history["t"].append(tt)
        self.access_history["forbidden_share_nodes"].append(float(forbidden_share))
        self.access_history["safe_lcc_share"].append(float(lcc_share))
        self.access_history["reachable_from_base_share"].append(float(reachable_share))

    def step(self):
        t = self.t
        self.log_access_metrics(t)

        # -------------------------
        # Spawn casualties
        # -------------------------
        if t < len(casualties_per_t):
            for cd in casualties_per_t[t]:
                c = Casualty(self)
                # --- enforce spawned triage from environment ---
                c.triage = str(cd.get("triage", getattr(c, "triage", "green")))
                c.current_triage = c.triage
                c.triage0 = c.triage

                # rebuild deterministic schedule to match chosen triage
                t0 = float(c.t_created)
                schedule = []
                cur = c.triage0
                t_abs = t0
                for _ in range(4):
                    if cur == "black":
                        break
                    dt = float(sample_max_delay(cur))
                    t_abs += dt
                    nxt = str(progress_triage_label(cur))
                    if nxt == cur:
                        break
                    schedule.append({"t": float(t_abs), "triage": nxt})
                    cur = nxt
                c.det_schedule = schedule

                # reset event log consistent with chosen triage
                c.triage_events = [{
                    "t": float(self.t),
                    "triage": str(c.current_triage),
                    "stab": float(sample_stabilization_time(c.current_triage)),
                }]
                c.stabilization_time_min = float(sample_stabilization_time(c.current_triage))

                c.x = cd["x"]
                c.y = cd["y"]
                c.state = "field"
                # mark if spawned in no-go (red tile)
                c.no_go = bool(is_no_go_at_time(self, c.x, c.y, self.t))
                # init unreachable flags (for unique counting)
                c._flag_spawn_no_go = False
                c._flag_filtered_no_go = False
                c._flag_amb_no_approach = False
                c._flag_amb_no_path = False
                c._flag_heli_line_blocked = False

                # spawned in no-go (unique)
                if c.no_go and (not c._flag_spawn_no_go):
                    c._flag_spawn_no_go = True
                    self.unreachable["spawned_in_no_go_unique"] += 1

                self.casualties.append(c)
                self.NUM_CASUALTIES += 1

        # -------------------------
        # Update casualty triage
        # -------------------------
        self.casualties.step()

        # -------------------------
        # Central coordination for dispatch and return decisions.
        # -------------------------
        self.coordinator.step()

        # -------------------------
        # Move rescue agents
        # -------------------------
        self.ambulances.step()
        self.helicopters.step()

        # -------------------------
        # Base treatment step
        # -------------------------
        self.base.step()

        # -------------------------
        # Logging
        # -------------------------
        self.history.append({
            "ambulances": [(a.x, a.y) for a in self.ambulances],
            "ambulances_speed": [a.speed for a in self.ambulances],
            "ambulances_cargo": [len(a.cargo) for a in self.ambulances],

            "helicopters": [(h.x, h.y) for h in self.helicopters],
            "helicopters_speed": [h.speed for h in self.helicopters],
            "helicopters_cargo": [len(h.cargo) for h in self.helicopters],

            "base": (self.base_x, self.base_y),

            "casualties": [
                (c.x, c.y, c.current_triage)
                for c in self.casualties if not c.evacuated
            ],
            "evacuated": sum(c.evacuated for c in self.casualties),

            "delivered_total": self.delivered_total,

            # Base statistics used for animation or plots.
            "base_queue": len(self.base.queue),
            "base_beds": len(self.base.in_treatment),
        })

    def end(self):
        # Default policy is whatever the model currently has (or SEVERITY fallback)
        bp = str(getattr(self, "BASE_POLICY", "SEVERITY")).upper()

        results = full_metrics_like_original(self, n_beds=int(self.N_BEDS), base_policy=bp)

        print("\n" + "="*40)
        print("CASEVAC SIMULATION METRICS")
        print("="*40)

        for k, v in results.items():
            if isinstance(v, float):
                print(f"{k:<35}: {v:.3f}")
            else:
                print(f"{k:<35}: {v}")

        print("="*40 + "\n")

        return results

class MetricsCollector:
    def __init__(self, model):
        self.model = model



    def compute(self):
        m = self.model
        C = list(m.casualties)
        N = len(C)

        def triage_at_abs(c, t_abs):
            label = c.triage_events[0]["triage"] if c.triage_events else c.triage
            for e in c.triage_events:
                if e["t"] <= t_abs:
                    label = e["triage"]
                else:
                    break
            return label

        # ----------------------------
        # F_t / S_t summary stats helper
        # ----------------------------
        def series_stats(x):
            """
            Return mean/min/max/var for a numeric series.
            Uses population variance (ddof=0). Returns np.nan if empty/missing.
            """
            if x is None:
                return {"mean": np.nan, "min": np.nan, "max": np.nan, "var": np.nan}

            x = np.asarray(x, dtype=float)
            if x.size == 0:
                return {"mean": np.nan, "min": np.nan, "max": np.nan, "var": np.nan}

            return {
                "mean": float(np.mean(x)),
                "min": float(np.min(x)),
                "max": float(np.max(x)),
                "var": float(np.var(x, ddof=0)),
            }

        F_stats = series_stats(getattr(m, "F_ts", None))
        S_stats = series_stats(getattr(m, "S_ts", None))


        # ----------------------------
        # groepen
        # ----------------------------
        rescued_to_base = [c for c in C if c.arrival_base_t is not None]
        picked_up = [c for c in C if c.pickup_time is not None]
        black = [c for c in C if getattr(c, "current_triage", None) == "black"]
        dead_in_queue = [c for c in C if getattr(c, "died_in_queue", False)]
        arrived = [c for c in C if c.arrival_base_t is not None]

        black_on_arrival = [
            c for c in arrived
            if triage_at_abs(c, c.arrival_base_t) == "black"
        ]

        black_during_transport = [
            c for c in arrived
            if (c.pickup_time is not None)
            and (triage_at_abs(c, c.pickup_time) != "black")
            and (triage_at_abs(c, c.arrival_base_t) == "black")
        ]


        # no-go stats (spawned or ever observed as no-go via coordinator flag)
        no_go_flagged = [c for c in C if bool(getattr(c, "no_go", False))]

        rescued_count = len(rescued_to_base)
        picked_count = len(picked_up)

        # ----------------------------
        # golden hour (60 min)
        # ----------------------------
        gh_arrival = [
            c for c in rescued_to_base
            if (c.arrival_base_t - c.t_created) <= 60
        ]
        gh_treatment = [
            c for c in rescued_to_base
            if c.t_bed is not None and (c.t_bed - c.t_created) <= 60
        ]

        # ----------------------------
        # tijden
        # ----------------------------
        pickup_times = [
            (c.pickup_time - c.t_created)
            for c in picked_up
            if c.pickup_time is not None
        ]
        transport_times = [
            (c.arrival_base_t - c.pickup_time)
            for c in rescued_to_base
            if c.pickup_time is not None and c.arrival_base_t is not None
        ]
        waiting_times = [
            (c.t_bed - c.arrival_base_t)
            for c in rescued_to_base
            if c.t_bed is not None and c.arrival_base_t is not None
        ]
        treatment_times = [
            (c.treatment_end_t - c.treatment_start_t)
            for c in rescued_to_base
            if c.treatment_start_t is not None and c.treatment_end_t is not None
        ]

        # ----------------------------
        # base util / queue stats
        # ----------------------------
        base = m.base
        beds_total_series = base.history.get("beds_total", [])
        queue_total_series = base.history.get("queue_total", [])

        avg_beds_used = float(np.mean(beds_total_series)) if beds_total_series else 0.0
        bed_utilization = (avg_beds_used / base.n_beds) if base.n_beds > 0 else np.nan

        avg_queue_len = float(np.mean(queue_total_series)) if queue_total_series else 0.0
        max_queue_len = int(max(queue_total_series)) if queue_total_series else 0

        # ----------------------------
        # per platform
        # ----------------------------
        picked_by_amb = [c for c in picked_up if c.picked_by == "ambulance"]
        picked_by_heli = [c for c in picked_up if c.picked_by == "helicopter"]

        # ----------------------------
        # return dict
        # ----------------------------
        return {
            "Total casualties spawned": N,

            "Picked up count": picked_count,
            "Picked up %": (picked_count / N) if N else 0.0,

            "Arrived at base count": rescued_count,
            "Arrived at base %": (rescued_count / N) if N else 0.0,

            "Black count (current)": len(black),
            "Black % (current)": (len(black) / N) if N else 0.0,
            "Black on arrival": len(black_on_arrival),
            "Black during transport": len(black_during_transport),
            "Died in queue": len(dead_in_queue),

            # no-go diagnostics
            "No-go flagged (current)": len(no_go_flagged),
            "No-go flagged % (current)": (len(no_go_flagged) / N) if N else 0.0,

            "Golden Hour % (Arrival to base)": (len(gh_arrival) / rescued_count) if rescued_count else 0.0,
            "Golden Hour % (Treatment start)": (len(gh_treatment) / rescued_count) if rescued_count else 0.0,

            "Avg pickup time (min)": float(np.mean(pickup_times)) if pickup_times else np.nan,
            "Avg transport time pickup->base (min)": float(np.mean(transport_times)) if transport_times else np.nan,
            "Mean waiting time in queue (min)": float(np.mean(waiting_times)) if waiting_times else np.nan,
            "Avg treatment time (min)": float(np.mean(treatment_times)) if treatment_times else np.nan,

            "Max queue length": max_queue_len,
            "Average queue length": avg_queue_len,

            "Average beds used": avg_beds_used,
            "Bed utilization": float(bed_utilization) if not np.isnan(bed_utilization) else np.nan,

            "Picked by ambulance": len(picked_by_amb),
            "Picked by helicopter": len(picked_by_heli),

            "Platform failures (ambulance)": int(m.platform_failures.get("ambulance", 0)),
            "Platform failures (helicopter)": int(m.platform_failures.get("helicopter", 0)),
            "Platform failures (total)": int(m.platform_failures.get("ambulance", 0) + m.platform_failures.get("helicopter", 0)),

            # ----------------------------
            # Fight situation series stats
            # ----------------------------
            "F_t mean": F_stats["mean"],
            "F_t min": F_stats["min"],
            "F_t max": F_stats["max"],
            "F_t variance": F_stats["var"],

            "S_t mean": S_stats["mean"],
            "S_t min": S_stats["min"],
            "S_t max": S_stats["max"],
            "S_t variance": S_stats["var"],
        }

#endregion

#region Base Replay Functions
# This block replays base-treatment outcomes for different bed capacities and queue policies using recorded arrivals.
def triage_at_time_from_schedule(triage0, det_schedule, t, t_freeze=None):
    """
    triage0: initial label at spawn
    det_schedule: [{"t": abs_time, "triage": label}, ...]
    t: time at which to evaluate
    t_freeze: if not None, triage stops progressing at t_freeze (treatment start)
    """
    if t_freeze is not None and t >= t_freeze:
        t = t_freeze

    label = str(triage0)
    for e in det_schedule:
        if float(e["t"]) <= float(t):
            label = str(e["triage"])
        else:
            break
    return label

def stab_for_triage_at_time(triage_events, t):
    cur = triage_events[0] if triage_events else {"stab": 0.0}
    for e in triage_events:
        if e["t"] <= t:
            cur = e
        else:
            break
    return float(cur.get("stab", 0.0))

def severity_rank(label):
    return {"red": 0, "yellow": 1, "green": 2}.get(label, 2)

def simulate_base_only(arrivals, n_beds, policy="SEVERITY"):
    """
    Base-only replay:
      - arrivals: list of dicts from model.base_arrivals
      - n_beds: int
      - policy: "SEVERITY" or "FIFO"
    Returns:
      dict with:
        t_bed: cid -> time treatment started
        t_end: cid -> time treatment ended
        died_in_queue: set(cid)
        dead_on_arrival: set(cid)
        t_series, queue_len_series, beds_used_series
    """

    # ---------- helpers ----------
    def triage_at_time(item, t):
        # ALWAYS use bed-independent schedule if present
        tri0 = item.get("triage0", item.get("triage", "green"))
        sched = item.get("det_schedule", [])
        if sched:
            label = str(tri0)
            for e in sched:
                if float(e.get("t", 0.0)) <= float(t):
                    label = str(e.get("triage", label))
                else:
                    break
            return label
        # fallback: if schedule missing
        return str(tri0)


    def stab_at_time(item, t):
        # Prefer stabilization time from triage_events if present
        evs = item.get("triage_events", None)
        if evs:
            cur = evs[0]
            for e in evs:
                if float(e.get("t", -1)) <= float(t):
                    cur = e
                else:
                    break
            return float(cur.get("stab", 0.0))
        # fallback: if you logged something else, else 0
        return float(item.get("stab", 0.0))

    def severity_rank(label):
        # lower is more severe
        return {"red": 0, "yellow": 1, "green": 2}.get(str(label), 2)

    # ---------- sort arrivals ----------
    arrivals_sorted = sorted(arrivals, key=lambda a: float(a.get("t_arrival", 0.0)))
    i = 0
    t = 0.0

    queue = []          # list of arrival dicts
    in_service = []     # min-heap of (t_finish, cid)

    t_bed = {}          # cid -> treatment start time
    t_end = {}          # cid -> treatment end time
    died_in_queue = set()
    dead_on_arrival = set()

    t_series = []
    queue_len_series = []
    beds_used_series = []

    # For strict FIFO we preserve insertion order; for SEVERITY we select each time.
    # We must also "clean" queue at each decision point: black-in-queue -> died_in_queue.
    def clean_queue(now):
        nonlocal queue
        newq = []
        for item in queue:
            cid = int(item.get("cid", -1))
            tri = triage_at_time(item, now)
            if tri == "black":
                died_in_queue.add(cid)
            else:
                newq.append(item)
        queue = newq

    def pick_from_queue(now):
        nonlocal queue
        clean_queue(now)
        if not queue:
            return None

        if str(policy).upper() == "FIFO":
            return queue.pop(0)

        # SEVERITY: recompute triage at 'now' for each queued patient
        best_idx = None
        best_key = None
        for idx, item in enumerate(queue):
            tri = triage_at_time(item, now)
            key = severity_rank(tri)
            if best_idx is None or key < best_key:
                best_idx = idx
                best_key = key
        return queue.pop(best_idx)

    # ---------- event loop ----------
    while i < len(arrivals_sorted) or in_service or queue:
        next_arrival_t = float(arrivals_sorted[i]["t_arrival"]) if i < len(arrivals_sorted) else float("inf")
        next_finish_t = float(in_service[0][0]) if in_service else float("inf")
        t_next = min(next_arrival_t, next_finish_t)

        if t_next == float("inf"):
            break

        t = float(t_next)

        # 1) finish treatments up to time t
        while in_service and float(in_service[0][0]) <= t + 1e-9:
            tf, cid = heapq.heappop(in_service)
            t_end[int(cid)] = float(tf)

        # 2) add arrivals at time t
        while i < len(arrivals_sorted) and float(arrivals_sorted[i]["t_arrival"]) <= t + 1e-9:
            item = arrivals_sorted[i]
            cid = int(item.get("cid", -1))

            tri_arr = triage_at_time(item, float(item["t_arrival"]))
            if tri_arr == "black":
                dead_on_arrival.add(cid)
            else:
                queue.append(item)

            i += 1

        # 3) start treatments if beds free
        while len(in_service) < int(n_beds):
            item = pick_from_queue(t)
            if item is None:
                break

            cid = int(item.get("cid", -1))
            tri_now = triage_at_time(item, t)

            # safety: if black at selection moment, count as died in queue
            if tri_now == "black":
                died_in_queue.add(cid)
                continue

            t_bed[cid] = float(t)
            stab = stab_at_time(item, t)
            heapq.heappush(in_service, (float(t) + float(stab), cid))

        # 4) log series
        # (clean first so queue length reflects "still alive in queue")
        clean_queue(t)
        t_series.append(float(t))
        queue_len_series.append(len(queue))
        beds_used_series.append(len(in_service))

    return {
        "t_bed": t_bed,
        "t_end": t_end,
        "died_in_queue": died_in_queue,
        "dead_on_arrival": dead_on_arrival,
        "t_series": t_series,
        "queue_len_series": queue_len_series,
        "beds_used_series": beds_used_series,
    }

def recompute_metrics_for_beds(arrivals, n_beds, base_policy="SEVERITY",
                               platform_failures=None):
    """
    arrivals: list of dicts from model.base_arrivals
    n_beds: int
    base_policy: "SEVERITY" or "FIFO"
    platform_failures: optional dict like {"ambulance":..., "helicopter":...}
    """

    # ---- totals (independent of beds) ----
    N_total = len(arrivals)  # NOTE: only includes those who ARRIVED at base if you only log in admit().
    # If you want "Total casualties spawned", you need separate total spawned count.
    # Easiest: store model.NUM_CASUALTIES separately and pass it in.

    # If you want to keep your original output definitions:
    # - Total casualties spawned: pass as argument from model.NUM_CASUALTIES
    # - Picked up count/%: pass from original run (or log all pickups)
    # - Arrived at base count/%: this arrival list size and % vs total spawned

    # ---- base-only replay for this n_beds ----
    rep = simulate_base_only(arrivals, n_beds=n_beds, policy=base_policy)
    t_bed = rep["t_bed"]
    t_end = rep["t_end"]
    died_in_queue = rep["died_in_queue"]
    dead_on_arrival = rep["dead_on_arrival"]

# queue deaths exclude black on arrival by construction now

    # ---- groups ----
    arrived = arrivals  # these are the ones that arrived at base by construction
    rescued_count = len(arrived)

    # "black count (current)" in your old code looked at *final current_triage* of all casualties.
    # Here we can approximate for arrived only:
    black_arrived = 0
    for a in arrived:
        # last triage event label is current at end of sim
        evs = a.get("triage_events", [])
        last_tri = evs[-1]["triage"] if evs else "green"
        if last_tri == "black":
            black_arrived += 1

    # ---- golden hour ----
    gh_arrival = 0
    gh_treatment = 0
    for a in arrived:
        cid = a["cid"]
        t_created = float(a["t_created"])
        t_arrival = float(a["t_arrival"])

        if (t_arrival - t_created) <= 60.0:
            gh_arrival += 1

        if cid in t_bed:
            if (float(t_bed[cid]) - t_created) <= 60.0:
                gh_treatment += 1

    # ---- times ----
    pickup_times = []
    transport_times = []
    waiting_times = []
    treatment_times = []

    for a in arrived:
        cid = a["cid"]
        t_created = float(a["t_created"])
        t_arrival = float(a["t_arrival"])
        ptime = a.get("pickup_time", None)

        if ptime is not None:
            pickup_times.append(float(ptime) - t_created)
            transport_times.append(t_arrival - float(ptime))

        if cid in t_bed:
            waiting_times.append(float(t_bed[cid]) - t_arrival)

        if cid in t_bed and cid in t_end:
            treatment_times.append(float(t_end[cid]) - float(t_bed[cid]))

    # ---- queue / bed series ----
    q_series = rep["queue_len_series"]
    b_series = rep["beds_used_series"]

    avg_queue_len = float(np.mean(q_series)) if q_series else 0.0
    max_queue_len = int(np.max(q_series)) if q_series else 0

    avg_beds_used = float(np.mean(b_series)) if b_series else 0.0
    bed_utilization = (avg_beds_used / float(n_beds)) if n_beds > 0 else np.nan

    # ---- picked_by counts (only among arrivals if logged) ----
    picked_by_amb = sum(1 for a in arrived if a.get("picked_by") == "ambulance")
    picked_by_heli = sum(1 for a in arrived if a.get("picked_by") == "helicopter")

    pf_amb = int(platform_failures.get("ambulance", 0)) if platform_failures else 0
    pf_heli = int(platform_failures.get("helicopter", 0)) if platform_failures else 0

    return {
        # These 3 depend on how you define totals. See note below.
        "Arrived at base count": rescued_count,

        "Black count (current)": black_arrived,
        "Black % (current)": (black_arrived / rescued_count) if rescued_count else 0.0,

        "Dead on arrival": len(dead_on_arrival),
        "Died in queue": len(died_in_queue),


        "Golden Hour % (Arrival to base)": (gh_arrival / rescued_count) if rescued_count else 0.0,
        "Golden Hour % (Treatment start)": (gh_treatment / rescued_count) if rescued_count else 0.0,

        "Avg pickup time (min)": float(np.mean(pickup_times)) if pickup_times else np.nan,
        "Avg transport time pickup->base (min)": float(np.mean(transport_times)) if transport_times else np.nan,
        "Mean waiting time in queue (min)": float(np.mean(waiting_times)) if waiting_times else np.nan,
        "Avg treatment time (min)": float(np.mean(treatment_times)) if treatment_times else np.nan,

        "Max queue length": max_queue_len,
        "Average queue length": avg_queue_len,

        "Average beds used": avg_beds_used,
        "Bed utilization": float(bed_utilization) if not np.isnan(bed_utilization) else np.nan,

        "Picked by ambulance": int(picked_by_amb),
        "Picked by helicopter": int(picked_by_heli),

        "Platform failures (ambulance)": pf_amb,
        "Platform failures (helicopter)": pf_heli,
        "Platform failures (total)": pf_amb + pf_heli,
    }

def full_metrics_like_original(model, n_beds, base_policy="SEVERITY"):

    arrivals = model.base_arrivals
    base_part = recompute_metrics_for_beds(
        arrivals=arrivals,
        n_beds=n_beds,
        base_policy=str(base_policy).upper(),   # replay policy
        platform_failures=getattr(model, "platform_failures", None)
    )

    total_spawned = int(model.NUM_CASUALTIES)

    # TOTAL black (includes field)
    black_total = sum(1 for c in model.casualties if getattr(c, "current_triage", None) == "black")

    picked_up_count = sum(1 for c in model.casualties if c.pickup_time is not None)
    arrived_count = len(arrivals)

    # -------------------------
    # Fight situation series stats (F_t, S_t)
    # -------------------------
    def series_stats(x):
        """
        Return mean/min/max/var for a numeric series.
        Uses population variance (ddof=0). Returns np.nan if empty/missing.
        """
        if x is None:
            return {"mean": np.nan, "min": np.nan, "max": np.nan, "var": np.nan}

        x = np.asarray(x, dtype=float)
        if x.size == 0:
            return {"mean": np.nan, "min": np.nan, "max": np.nan, "var": np.nan}

        return {
            "mean": float(np.mean(x)),
            "min": float(np.min(x)),
            "max": float(np.max(x)),
            "var": float(np.var(x, ddof=0)),
        }

    F_stats = series_stats(getattr(model, "F_ts", None))
    S_stats = series_stats(getattr(model, "S_ts", None))


    out = {
        "Total casualties spawned": total_spawned,

        "Picked up count": picked_up_count,
        "Picked up %": (picked_up_count / total_spawned) if total_spawned else 0.0,

        "Arrived at base count": arrived_count,
        "Arrived at base %": (arrived_count / total_spawned) if total_spawned else 0.0,

        # total black (field + anything else)
        "Black count (current, total)": int(black_total),
        "Black % (current, total)": (black_total / total_spawned) if total_spawned else 0.0,

        # -------------------------
        # Fight situation series stats
        # -------------------------
        "F_t mean": F_stats["mean"],
        "F_t min": F_stats["min"],
        "F_t max": F_stats["max"],
        "F_t variance": F_stats["var"],

        "S_t mean": S_stats["mean"],
        "S_t min": S_stats["min"],
        "S_t max": S_stats["max"],
        "S_t variance": S_stats["var"],
    }

    out["Mission aborted: black on arrival"] = int(model.mission_aborts.get("black_on_arrival", 0))
    out["Mission aborted: no-go block"] = int(model.mission_aborts.get("no_go_block", 0))
    out["Mission aborted: target became no-go"] = int(model.mission_aborts.get("target_became_no_go", 0))
    out["Mission aborted: target became black before pickup"] = int(model.mission_aborts.get("target_became_black_before_pickup", 0))

    # merge base-dependent
    out.update(base_part)

    # derived: black who never arrived (field / not-arrived)
    if "Black count (current)" in out:  # your arrived-only black
        out["Black not arrived (derived)"] = int(out["Black count (current, total)"] - out["Black count (current)"])

    # -------------------------
    # Tile flip metrics summary (seed-level)
    # -------------------------
    flip_hist = getattr(model, "tile_flip_history", []) or []
    if flip_hist:
        total_seed_flips = int(sum(h.get("total_flips", 0) for h in flip_hist))

        agg = Counter()
        for h in flip_hist:
            agg.update(h.get("counts", {}))

        out["Tile flips (seed-level, total)"] = total_seed_flips

        for k in ["0→1", "1→0", "0→2", "2→0", "1→2", "2→1"]:
            out[f"Tile flips (seed) {k}"] = int(agg.get(k, 0))

        out["Tile flips (seed-level, avg/tick)"] = float(total_seed_flips / max(1, len(flip_hist)))
        out["Tile flips (seed-level, max/tick)"] = int(max(h.get("total_flips", 0) for h in flip_hist))
    else:
        out["Tile flips (seed-level, total)"] = 0

    # -------------------------
    # Access loss summaries (tiles + network)
    # -------------------------
    tile_sh = getattr(model, "tile_share_history", []) or []
    if tile_sh:
        enemy_series = [h.get("share_enemy", 0.0) for h in tile_sh]
        contested_series = [h.get("share_contested", 0.0) for h in tile_sh]
        out["Tile share enemy (avg)"] = float(np.mean(enemy_series))
        out["Tile share enemy (max)"] = float(np.max(enemy_series))
        out["Tile share contested (avg)"] = float(np.mean(contested_series))
    else:
        out["Tile share enemy (avg)"] = np.nan
        out["Tile share enemy (max)"] = np.nan
        out["Tile share contested (avg)"] = np.nan

    acc = getattr(model, "access_history", None)
    if acc and acc.get("forbidden_share_nodes"):
        out["Forbidden nodes share (avg)"] = float(np.mean(acc["forbidden_share_nodes"]))
        out["Forbidden nodes share (max)"] = float(np.max(acc["forbidden_share_nodes"]))
        out["Reachable-from-base share (avg)"] = float(np.mean(acc["reachable_from_base_share"]))
        out["Safe LCC share (min)"] = float(np.min(acc["safe_lcc_share"]))
    else:
        out["Forbidden nodes share (avg)"] = np.nan
        out["Forbidden nodes share (max)"] = np.nan
        out["Reachable-from-base share (avg)"] = np.nan
        out["Safe LCC share (min)"] = np.nan

    # -------------------------
    # Unreachable casualties summaries
    # -------------------------
    unr = getattr(model, "unreachable", {}) or {}
    out["Unreachable: spawned in no-go (unique)"] = int(unr.get("spawned_in_no_go_unique", 0))
    out["Unreachable: filtered no-go current (unique)"] = int(unr.get("filtered_no_go_current_unique", 0))
    out["Unreachable: amb no-approach (unique)"] = int(unr.get("amb_no_approach_unique", 0))
    out["Unreachable: amb no-path (unique)"] = int(unr.get("amb_no_path_unique", 0))
    out["Unreachable: heli line blocked (unique)"] = int(unr.get("heli_line_blocked_unique", 0))

    out["Unreachable: filtered no-go current (freq)"] = int(unr.get("filtered_no_go_current_freq", 0))
    out["Unreachable: amb no-approach (freq)"] = int(unr.get("amb_no_approach_freq", 0))
    out["Unreachable: amb no-path (freq)"] = int(unr.get("amb_no_path_freq", 0))
    out["Unreachable: heli line blocked (freq)"] = int(unr.get("heli_line_blocked_freq", 0))

    return out

#endregion

#region MC config
# This block defines the Monte Carlo experiment grid, output locations, and scenario parameters.

OUT_DIR = "mc_results"
STATE_PATH = os.path.join(OUT_DIR, "mc_state.json")

N_RUNS = 99
START_SEED = 1

AMB_GRID = [4, 8, 12, 16]
HELI_GRID = [0, 1, 2, 3]

HEURISTIC_GRID = [1, 2, 3, 4, 5, 6]                  # dispatch heuristic vergelijken
COORD_GRID = [False]               # True voor clairvoyant myopic vs clairvoyant
DOCTRINE_GRID_MYOPIC = ["SOFT", "HARD"]  # myopic: beide
DOCTRINE_GRID_CLAIR = ["SOFT"]           # clairvoyant: alleen SOFT

BASE_POLICY_GRID = ["SEVERITY", "FIFO"]  # via base replay, dus geen reruns

BED_GRID = [2, 4, 6, 8, 10, 12, 15, 20]

os.makedirs(OUT_DIR, exist_ok=True)

RESULTS_PATH = os.path.join(OUT_DIR, "casevac_mc_results.ndjson")
ERRORS_PATH  = os.path.join(OUT_DIR, "casevac_mc_errors.ndjson")
PROGRESS_LOG = os.path.join(OUT_DIR, "casevac_mc_progress.log")



#endregion

#region MC Helpers
# This block contains helper functions for scenario hashing, output persistence, and reusable Monte Carlo setup.

def build_environment_for_seed(seed: int):
    """
    Bouwt de volledige environment + spawn schedule voor 1 seed (CRN).
    Zet alles in globals zodat CasevacModel.setup() exact dezelfde omgeving ziet
    voor alle scenario’s binnen deze seed.
    """

    # -------------------------
    # RNG reset (CRN)
    # -------------------------
    random.seed(seed)
    np.random.seed(seed)

    # The frontline angle must also depend on the seed.
    global frontline_angle
    frontline_angle = np.random.uniform(0, 360)

    # -------------------------
    # Voronoi tiles
    # -------------------------
    global tile_map_init, tile_class_init, seeds, nearest, tree_seeds, frontline_state, cells_per_seed

    tile_map_init, tile_class_init, seeds, nearest, tree_seeds, frontline_state, cells_per_seed = generate_tiles(
        x=x, y=y,
        x_min=x_min, y_min=y_min,
        Dx=Dx, Dy=Dy,
        N_seeds=N_seeds,
        frontline_angle_deg=frontline_angle,
        seed=seed
    )

    V = tile_class_init.copy()
    V_dyn = V.copy()

    # -------------------------
    # Fight dynamics
    # -------------------------
    global F_ts, S_ts
    F_ts, S_ts = simulate_fight_situation(TIME_STEPS, seed=seed)

    # -------------------------
    # Initial hotspots
    # -------------------------
    hotspots = []
    orange_seeds = np.where(V == 1)[0]
    M0 = np.random.poisson(LAMBDA_0)

    for _ in range(M0):
        if len(orange_seeds) == 0:
            break
        sidx = np.random.choice(orange_seeds)
        hotspots.append(
            create_hotspot(
                sidx, seeds, cells_per_seed,
                Dx=Dx, Dy=Dy, x=x, y=y,
                amp_range=(AMP_MIN, AMP_MAX)
            )
        )

    # -------------------------
    # Storage (globals!)
    # -------------------------
    global tile_snapshots, hotspot_grids, casualties_per_t, tile_share_history
    global tile_flip_history, tile_flip_totals

    tile_snapshots = []
    hotspot_grids = []
    casualties_per_t = []
    tile_share_history = []

    tile_flip_history = []
    tile_flip_totals = []

    casualty_id_seq = 0

    # =========================
    # MAIN ENVIRONMENT LOOP
    # =========================
    for t in range(TIME_STEPS):
        F_t, S_t = F_ts[t], S_ts[t]

        # ---- Tiles update
        V_prev = V_dyn.copy()

        update_frontline_delta(
            frontline_state,
            F_t, S_t,
            F_ref=F_high,
            F_max=F_max,
            v_max=60.0,
            s_scale=1.0,
            delta_max=8000.0
        )

        V_dyn = reclassify_seeds_from_frontline(seeds, frontline_state)

        # ---- Flip metrics
        total_flips_t, flip_counts_t, flip_mat_t = compute_flip_counts(
            V_prev, V_dyn, classes=(0, 1, 2)
        )

        tile_flip_history.append({
            "t": int(t),
            "total_flips": int(total_flips_t),
            "counts": dict(flip_counts_t),
            "matrix": flip_mat_t.copy(),
            "delta": float(frontline_state.get("delta", 0.0)),
            "F_t": float(F_t),
            "S_t": float(S_t),
        })
        tile_flip_totals.append(int(total_flips_t))

        # ---- Hotspots from flips
        flipped_to_orange = np.where((V_prev != 1) & (V_dyn == 1))[0]
        for sidx in flipped_to_orange:
            hotspots.append(
                create_hotspot(
                    sidx, seeds, cells_per_seed,
                    Dx=Dx, Dy=Dy, x=x, y=y,
                    amp_range=(A_TILE_ON_FLIP_MIN, A_TILE_ON_FLIP_MAX)
                )
            )

        # ---- Random hotspot spawn
        orange_seed_indices = np.where(V_dyn == 1)[0]
        for sidx in orange_seed_indices:
            if np.random.rand() < BASE_SPAWN_PROB:
                hotspots.append(
                    create_hotspot(
                        sidx, seeds, cells_per_seed,
                        Dx=Dx, Dy=Dy, x=x, y=y,
                        amp_range=(AMP_MIN, AMP_MAX)
                    )
                )

        # ---- Remove invalid hotspots
        hotspots = [h for h in hotspots if V_dyn[int(h["seed_idx"])] == 1]

        if F_t < F_low:
            k_t = P0_LOW + BETA_LOW * (F_low - F_t)
            k_t = np.clip(k_t, 0.0, 1.0)
            hotspots = [h for h in hotspots if np.random.rand() > k_t]

        # ---- Render hotspots
        H_grid = np.zeros_like(X, dtype=float)
        orange_mask_grid = (V_dyn[nearest].reshape(X.shape) == 1)

        for h in hotspots:
            stamp_hotspot_into_grid(
                H_grid, h,
                x_min=x_min, y_min=y_min,
                Dx=Dx, Dy=Dy,
                tile_mask=orange_mask_grid
            )

        if H_grid.max() > 0:
            H_grid /= H_grid.max()

        # ---- Casualty generation
        C_t = casualty_at_t(F_t, S_t, xi=xi_casualty, Lambda_c=Lambda_c)
        casualties_this_t = []

        if C_t > 0:
            probs = H_grid.ravel().astype(float)
            if probs.sum() <= 0:
                probs = np.ones_like(probs)
            probs /= probs.sum()

            sampled_idxs = np.random.choice(len(probs), size=C_t, replace=True, p=probs)

            for idx in sampled_idxs:
                iy, ix = divmod(int(idx), X.shape[1])
                cx = x[ix] + (np.random.rand() - 0.5) * Dx
                cy = y[iy] + (np.random.rand() - 0.5) * Dy
                triage = str(sample_triage_label())

                cd = {
                    "id": int(casualty_id_seq),
                    "t": int(t),
                    "grid_idx": int(idx),
                    "x": float(cx),
                    "y": float(cy),
                    "triage": triage,
                    "current_triage": triage,
                    "stabilization_time_min": float(sample_stabilization_time(triage)),
                    "max_tolerable_delay_min": float(sample_max_delay(triage)),
                    "triage_history": [{"t": int(t), "triage": triage}],
                    "evacuated": False,
                    "pickup_time": None,
                    "dropoff_time": None,
                    "rescued": False,
                    "arrival_base_t": None,
                    "treatment_start_t": None,
                    "treatment_end_t": None,
                    "stabilized": False,
                }
                casualty_id_seq += 1
                casualties_this_t.append(cd)

        # ---- Store snapshots (ALWAYS)
        casualties_per_t.append(casualties_this_t)
        tile_grid_t = V_dyn[nearest].reshape(X.shape)
        tile_snapshots.append(tile_grid_t)
        hotspot_grids.append(H_grid)

        tile_share_history.append({
            "t": int(t),
            "share_friendly": float(np.mean(tile_grid_t == 0)),
            "share_contested": float(np.mean(tile_grid_t == 1)),
            "share_enemy": float(np.mean(tile_grid_t == 2)),
        })

    # -------------------------
    # Return env summary (logging only)
    # -------------------------
    return {
        "seed": int(seed),
        "frontline_angle": float(frontline_angle),
        "time_steps": int(TIME_STEPS),
        "total_tile_flips": int(sum(tile_flip_totals)) if tile_flip_totals else 0,
        "total_spawned_env": int(sum(len(lst) for lst in casualties_per_t)),
    }

def log_line(msg: str):
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    s = f"[{ts}] {msg}"
    print(s)
    with open(PROGRESS_LOG, "a", encoding="utf-8") as f:
        f.write(s + "\n")
        f.flush()

def key_for(seed, amb, heli, heuristic, use_clairvoyant, doctrine, base_policy, bed):
    return (
        int(seed),
        int(amb),
        int(heli),
        int(heuristic),
        bool(use_clairvoyant),
        str(doctrine).upper(),
        str(base_policy).upper(),
        int(bed),
    )

def load_completed_keys(results_path: str):
    """
    Resume-safe: leest NDJSON en bouwt set met keys.
    Backwards compatible: als oude runs geen heuristic/base_policy hadden,
    dan zetten we defaults zodat het niet crasht (maar idealiter start je clean).
    """
    done = set()
    if not os.path.exists(results_path):
        return done

    with open(results_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)

                heuristic = obj.get("DISPATCH_HEURISTIC", obj.get("dispatch_heuristic", 1))
                base_pol  = obj.get("BASE_POLICY", obj.get("Base_Policy", obj.get("base_policy", "SEVERITY")))

                done.add(key_for(
                    obj["seed"],
                    obj["NUM_AMBULANCES"],
                    obj["NUM_HELICOPTERS"],
                    heuristic,
                    obj["USE_CLAIRVOYANT_COORDINATOR"],
                    obj["PLATFORM_DOCTRINE"],
                    base_pol,
                    obj["N_BEDS"],
                ))
            except Exception:
                continue
    return done

def save_state(state: dict, retries: int = 8, delay_s: float = 0.15):
    """
    OneDrive/AV lock-proof-ish.
    """
    os.makedirs(OUT_DIR, exist_ok=True)
    tmp = STATE_PATH + ".tmp"

    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=2)
        f.flush()
        os.fsync(f.fileno())

    last_err = None
    for attempt in range(retries):
        try:
            os.replace(tmp, STATE_PATH)
            return
        except PermissionError as e:
            last_err = e
            time.sleep(delay_s)

    fallback = os.path.join(OUT_DIR, f"mc_state_fallback_{int(time.time())}.json")
    with open(fallback, "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=2)
        f.flush()
        os.fsync(f.fileno())

    try:
        if os.path.exists(tmp):
            os.remove(tmp)
    except Exception:
        pass

    log_line(f"WARNING: could not replace state file due to lock: {last_err}. Wrote fallback: {fallback}")

def write_ndjson(path: str, obj: dict):
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")
        f.flush()
#endregion

#region MC LOOP
# This block runs the Monte Carlo experiment, skips completed scenarios, and writes scenario summaries.
def run_mc():
    log_line(f"Starting MC (resume-enabled): N_RUNS={N_RUNS}, START_SEED={START_SEED}")
    log_line(f"Results: {RESULTS_PATH}")
    log_line(f"Errors : {ERRORS_PATH}")
    log_line(f"State  : {STATE_PATH}")

    done = load_completed_keys(RESULTS_PATH)
    log_line(f"Loaded completed entries: {len(done)}")

    # Scenario count per seed; the doctrine grid differs per coordinator setting.
    total_scenarios = 0
    for _amb in AMB_GRID:
        for _heli in HELI_GRID:
            for _heur in HEURISTIC_GRID:
                for _clair in COORD_GRID:
                    doctrines = DOCTRINE_GRID_CLAIR if _clair else DOCTRINE_GRID_MYOPIC
                    for _doc in doctrines:
                        total_scenarios += 1

    log_line(f"Scenarios per seed (filtered): {total_scenarios}")
    log_line(f"Per scenario we compute: {len(BASE_POLICY_GRID)} base policies x {len(BED_GRID)} bed levels via REPLAY (no reruns).")

    try:
        for i in range(N_RUNS):
            seed = START_SEED + i
            log_line(f"=== SEED {seed} ({i+1}/{N_RUNS}) ===")

            save_state({"status": "building_env", "seed": int(seed), "run_index": int(i)})

            # 1) Build CRN environment once per seed
            try:
                env_info = build_environment_for_seed(seed)
                log_line(f"Env built. spawned={env_info['total_spawned_env']} flips={env_info['total_tile_flips']}")
            except Exception as e:
                err = {
                    "where": "build_environment_for_seed",
                    "seed": int(seed),
                    "error": str(e),
                    "traceback": traceback.format_exc(),
                }
                write_ndjson(ERRORS_PATH, err)
                log_line(f"ENV ERROR seed={seed}: {e} (skipping seed)")
                continue

            scen_idx = 0

            for amb in AMB_GRID:
                for heli in HELI_GRID:
                    for heuristic in HEURISTIC_GRID:
                        for use_clairvoyant in COORD_GRID:

                            doctrines = DOCTRINE_GRID_CLAIR if use_clairvoyant else DOCTRINE_GRID_MYOPIC

                            for doctrine in doctrines:
                                scen_idx += 1
                                scen_name = (
                                    f"amb={amb} heli={heli} heur={heuristic} "
                                    f"coord={'clair' if use_clairvoyant else 'myopic'} doc={doctrine}"
                                )
                                log_line(f"[seed {seed}] Scenario {scen_idx}/{total_scenarios}: {scen_name}")

                                # 2) Determine which (policy, bed) combos are missing
                                missing = []
                                for base_policy in BASE_POLICY_GRID:
                                    for b in BED_GRID:
                                        k = key_for(seed, amb, heli, heuristic, use_clairvoyant, doctrine, base_policy, b)
                                        if k not in done:
                                            missing.append((str(base_policy).upper(), int(b)))

                                if not missing:
                                    log_line(f"[seed {seed}] Scenario already complete -> skipped.")
                                    continue

                                save_state({
                                    "status": "running_scenario",
                                    "seed": int(seed),
                                    "run_index": int(i),
                                    "NUM_AMBULANCES": int(amb),
                                    "NUM_HELICOPTERS": int(heli),
                                    "DISPATCH_HEURISTIC": int(heuristic),
                                    "USE_CLAIRVOYANT_COORDINATOR": bool(use_clairvoyant),
                                    "PLATFORM_DOCTRINE": str(doctrine).upper(),
                                    "missing": [{"BASE_POLICY": bp, "N_BEDS": b} for (bp, b) in missing],
                                })

                                try:
                                    # 3) Run AgentPy model exactly once for this scenario
                                    globals()["NUM_AMBULANCES"] = int(amb)
                                    globals()["NUM_HELICOPTERS"] = int(heli)
                                    globals()["DISPATCH_HEURISTIC"] = int(heuristic)
                                    globals()["USE_CLAIRVOYANT_COORDINATOR"] = bool(use_clairvoyant)
                                    globals()["PLATFORM_DOCTRINE"] = str(doctrine).upper()

                                    # CRN: keep tie-break randomness stable per scenario as well
                                    random.seed(seed)
                                    np.random.seed(seed)
                                    globals()["F_ts"] = F_ts
                                    globals()["S_ts"] = S_ts
                                    model = CasevacModel()
                                    model.run(steps=TIME_STEPS)

                                    # 4) Replay base outcomes for each missing (policy, bed)
                                    for (base_policy, b) in missing:
                                        # Pass base_policy into the replay-based metrics function.
                                        res = full_metrics_like_original(
                                            model,
                                            n_beds=int(b),
                                            base_policy=str(base_policy).upper()
                                        )

                                        out = {
                                            "seed": int(seed),
                                            "run_index": int(i),

                                            "NUM_AMBULANCES": int(amb),
                                            "NUM_HELICOPTERS": int(heli),
                                            "DISPATCH_HEURISTIC": int(heuristic),

                                            "USE_CLAIRVOYANT_COORDINATOR": bool(use_clairvoyant),
                                            "PLATFORM_DOCTRINE": str(doctrine).upper(),

                                            "BASE_POLICY": str(base_policy).upper(),
                                            "N_BEDS": int(b),

                                            "frontline_angle": float(env_info["frontline_angle"]),
                                            "total_spawned_env": int(env_info["total_spawned_env"]),
                                            "total_tile_flips_env": int(env_info["total_tile_flips"]),

                                            "metrics": res,
                                        }

                                        write_ndjson(RESULTS_PATH, out)
                                        done.add(key_for(seed, amb, heli, heuristic, use_clairvoyant, doctrine, base_policy, b))

                                        save_state({
                                            "status": "wrote_combo",
                                            "seed": int(seed),
                                            "run_index": int(i),
                                            "NUM_AMBULANCES": int(amb),
                                            "NUM_HELICOPTERS": int(heli),
                                            "DISPATCH_HEURISTIC": int(heuristic),
                                            "USE_CLAIRVOYANT_COORDINATOR": bool(use_clairvoyant),
                                            "PLATFORM_DOCTRINE": str(doctrine).upper(),
                                            "BASE_POLICY": str(base_policy).upper(),
                                            "N_BEDS": int(b),
                                        })

                                except KeyboardInterrupt:
                                    raise
                                except Exception as e:
                                    err = {
                                        "where": "scenario_run_or_replay",
                                        "seed": int(seed),
                                        "NUM_AMBULANCES": int(amb),
                                        "NUM_HELICOPTERS": int(heli),
                                        "DISPATCH_HEURISTIC": int(heuristic),
                                        "USE_CLAIRVOYANT_COORDINATOR": bool(use_clairvoyant),
                                        "PLATFORM_DOCTRINE": str(doctrine).upper(),
                                        "missing": missing,
                                        "error": str(e),
                                        "traceback": traceback.format_exc(),
                                    }
                                    write_ndjson(ERRORS_PATH, err)
                                    log_line(f"SCENARIO ERROR seed={seed} {scen_name}: {e} (continuing)")
                                    continue

            log_line(f"=== SEED {seed} DONE ===")

        save_state({"status": "done"})
        log_line("MC finished.")

    except KeyboardInterrupt:
        save_state({"status": "stopped_by_user"})
        log_line("Stopped by user (Ctrl+C). You can restart and it will resume automatically.")


# Run the Monte Carlo experiment.
run_mc()
#endregion


