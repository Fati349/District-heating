"""
District Heating Network Optimization
Iterative Local Search (ILS) – Metaheuristic
IMT Atlantique – Networked Systems / Operational Research

Problem
-------
Select N-1 directed edges from a complete graph of N nodes to form a
spanning tree rooted at the source node.
Minimise total annual expenses = costs - revenue, subject to:
  - Tree structure (N-1 edges, connected, no cycle)
  - Unidirectionality per undirected edge
  - Thermal flow equations (P_in / P_out balance with losses)
  - Pipe capacity constraints
  - Source generation capacity constraint

ILS Design (following lecture03 guidelines)
-------------------------------------------
(1) Solution representation
    A list of N-1 directed edges [(i,j), ...] forming a valid
    spanning tree rooted at `source`.

(2) Greedy initial solution (constructive heuristic)
    Prim-like: starting from source, always add the cheapest feasible
    edge (checked against pipe/source capacity constraints).

(3) Local Search - intensification (two MOVE operators)
    - 2-opt-Swap:      Remove one tree edge, reconnect the two resulting
                      subtrees with a different non-tree edge.
    - Reorientation:  Reverse the direction of one tree edge.
    Both applied alternately; first-improvement strategy.

(4) Perturbation - diversification (Double-Bridge style)
    Remove one random tree edge, reconnect with a randomly chosen
    valid replacement edge to escape the current local optimum.

(5) Acceptance criterion
    Accept the new local optimum only if it strictly improves the
    current local optimum (greedy acceptance).
    Track the global best separately.

Usage
-----
    python ILS_district_heating.py
"""

import json
import math
import random
import os
import time
from collections import defaultdict


# =============================================================================
# 1.  DATA LOADING
# =============================================================================

def load_data(filename: str) -> dict:
    """Load and pre-process instance data from a JSON file."""
    with open(filename) as f:
        raw = json.load(f)

    N      = int(raw["Nodes"])
    source = int(raw["SourceNum"]) - 1      # convert 1-indexed to 0-indexed
    coords = raw["NodesCord"]

    # Euclidean distances
    l = [[math.sqrt((coords[i][0] - coords[j][0]) ** 2 +
                    (coords[i][1] - coords[j][1]) ** 2)
          for j in range(N)] for i in range(N)]

    return {
        "N"         : N,
        "source"    : source,
        "coords"    : coords,
        "l"         : l,
        "theta_fix" : raw["vfix(thetaijfix)"],
        "theta_var" : raw["vvar(thetaijvar)"],
        "c_fix"     : raw["FixedUnitCost"],
        "c_var"     : raw["cvar(cijvar)"],
        "c_heat"    : raw["cheat(ciheat)"],
        "c_om"      : raw["com(cijom)"],
        "c_rev"     : raw["crev(cijrev)"],
        "T_flh"     : raw["Tflh(Tiflh)"],
        "beta"      : raw["Betta"],
        "lam"       : raw["Lambda"],
        "alpha"     : raw["Alpha"],
        "d"         : raw["EdgesDemandPeak(dij)"],
        "D"         : raw["EdgesDemandAnnual(Dij)"],
        "C_max"     : raw["Cmax(cijmax)"],
        "Q_max"     : raw["SourceMaxCap(Qimax)"][source],
        "p_umd"     : raw["pumd(pijumd)"],
    }


# =============================================================================
# 2.  FLOW COMPUTATION
# =============================================================================

def compute_flows(tree_edges: list, data: dict):
    """
    Compute thermal power flows for a directed spanning tree (bottom-up).

    For each edge (i,j) in post-order (leaves first):
        eta[i,j]   = 1 - theta_var[i,j] * l[i,j]
        delta[i,j] = d[i,j]*beta*lam + theta_fix[i,j]*l[i,j]
        P_in[i,j]  = (need_out[j] + delta[i,j]) / eta[i,j]

    where need_out[j] = sum of P_in[j,k] for all children k of j.

    Returns (P_in dict, total_source_output) or None if infeasible.
    """
    N         = data["N"]
    source    = data["source"]
    d         = data["d"]
    theta_fix = data["theta_fix"]
    theta_var = data["theta_var"]
    l         = data["l"]
    beta      = data["beta"]
    lam       = data["lam"]
    C_max     = data["C_max"]
    Q_max     = data["Q_max"]

    children = defaultdict(list)
    parent   = {}
    for (a, b) in tree_edges:
        children[a].append(b)
        parent[b] = a

    # Post-order traversal
    visited = set()
    order   = []
    stack   = [source]
    while stack:
        node = stack[-1]
        if node not in visited:
            visited.add(node)
            for ch in children[node]:
                stack.append(ch)
        else:
            stack.pop()
            order.append(node)

    need_out = {n: 0.0 for n in range(N)}
    P_in     = {}

    for node in order:
        if node == source:
            continue
        par = parent[node]
        i, j = par, node

        eta   = 1.0 - theta_var[i][j] * l[i][j]
        delta = d[i][j] * beta * lam + theta_fix[i][j] * l[i][j]
        p_in  = (need_out[j] + delta) / eta

        if p_in > C_max[i][j] + 1e-9:   # pipe capacity violated
            return None

        P_in[(i, j)] = p_in
        need_out[par] += p_in

    if need_out[source] > Q_max + 1e-9:  # source capacity violated
        return None

    return P_in, need_out[source]


# =============================================================================
# 3.  OBJECTIVE FUNCTION
# =============================================================================

def evaluate(tree_edges: list, data: dict) -> float:
    """
    Compute total annual expenses Z (euros/year):

    Z = HeatGenerationCost + MaintenanceCost
      + FixedInvestmentCost + VariableInvestmentCost
      + UnmetDemandPenalty  - TotalRevenue

    Returns +inf if the solution is infeasible.
    """
    N      = data["N"]
    source = data["source"]
    l      = data["l"]
    D      = data["D"]
    lam    = data["lam"]
    c_rev  = data["c_rev"]
    c_heat = data["c_heat"]
    T_flh  = data["T_flh"]
    beta   = data["beta"]
    c_om   = data["c_om"]
    c_fix  = data["c_fix"]
    c_var  = data["c_var"]
    alpha  = data["alpha"]
    p_umd  = data["p_umd"]

    result = compute_flows(tree_edges, data)
    if result is None:
        return float("inf")

    P_in, total_source_out = result

    constructed = set()
    for (i, j) in tree_edges:
        constructed.add((min(i, j), max(i, j)))

    revenue     = sum(D[i][j] * lam * c_rev[i][j]           for (i,j) in tree_edges)
    heat_gen    = total_source_out * T_flh * c_heat[source] / beta
    maintenance = sum(l[i][j] * c_om[i][j]                  for (i,j) in tree_edges)
    fixed_inv   = alpha * sum(l[i][j] * c_fix                for (i,j) in tree_edges)
    var_inv     = alpha * sum(P_in[(i,j)] * l[i][j] * c_var[i][j] for (i,j) in tree_edges)
    unmet       = sum(D[i][j] * p_umd[i][j]
                      for i in range(N) for j in range(i+1, N)
                      if (i, j) not in constructed)

    return heat_gen + maintenance + fixed_inv + var_inv + unmet - revenue


# =============================================================================
# 4.  TREE STRUCTURE HELPERS
# =============================================================================

def is_valid_tree(tree_edges: list, data: dict) -> bool:
    """Check that edges form a valid directed spanning tree rooted at source."""
    N      = data["N"]
    source = data["source"]

    if len(tree_edges) != N - 1:
        return False

    in_degree = defaultdict(int)
    children  = defaultdict(list)
    for (a, b) in tree_edges:
        in_degree[b] += 1
        children[a].append(b)

    for node in range(N):
        expected = 0 if node == source else 1
        if in_degree[node] != expected:
            return False

    visited = {source}
    queue   = [source]
    while queue:
        node = queue.pop()
        for ch in children[node]:
            if ch not in visited:
                visited.add(ch)
                queue.append(ch)

    return len(visited) == N


def get_subtree(removed_edge: tuple, tree_edges: list, data: dict) -> set:
    """
    Return nodes in the subtree rooted at removed_edge[1]
    after removing `removed_edge` from the tree.
    """
    children = defaultdict(list)
    for (a, b) in tree_edges:
        if (a, b) != removed_edge:
            children[a].append(b)

    subtree = set()
    queue   = [removed_edge[1]]
    while queue:
        node = queue.pop()
        subtree.add(node)
        for ch in children[node]:
            queue.append(ch)
    return subtree


# =============================================================================
# 5.  GREEDY INITIAL SOLUTION  (constructive heuristic)
# =============================================================================

def greedy_initial_solution(data: dict) -> list:
    """
    Capacity-aware Prim-like greedy construction:

    Starting from the source, at each step add the edge (in_tree -> new_node)
    with the lowest greedy cost that keeps the partial tree feasible.

    Greedy cost = alpha * l[i][j] * c_fix        (annualised fixed investment)
               + l[i][j] * c_om[i][j]            (O&M cost)
               - D[i][j] * lam * c_rev[i][j]     (expected annual revenue)
    """
    N      = data["N"]
    source = data["source"]
    l      = data["l"]
    D      = data["D"]
    lam    = data["lam"]
    c_rev  = data["c_rev"]
    c_om   = data["c_om"]
    c_fix  = data["c_fix"]
    alpha  = data["alpha"]

    in_tree    = {source}
    tree_edges = []

    for _ in range(N - 1):
        best_cost = float("inf")
        best_edge = None

        for i in in_tree:
            for j in range(N):
                if j in in_tree:
                    continue
                candidate = tree_edges + [(i, j)]
                if compute_flows(candidate, data) is not None:
                    g = (alpha * l[i][j] * c_fix
                         + l[i][j] * c_om[i][j]
                         - D[i][j] * lam * c_rev[i][j])
                    if g < best_cost:
                        best_cost = g
                        best_edge = (i, j)

        # Fallback (should rarely trigger): any connecting edge
        if best_edge is None:
            for i in in_tree:
                for j in range(N):
                    if j not in in_tree:
                        best_edge = (i, j)
                        break
                if best_edge:
                    break

        tree_edges.append(best_edge)
        in_tree.add(best_edge[1])

    return tree_edges


# =============================================================================
# 6.  LOCAL SEARCH  (intensification)
# =============================================================================

def move_2opt_swap(tree_edges: list, data: dict,
                   current_cost: float) -> tuple:
    """
    Edge-Swap MOVE (intensification operator):

    Removing tree edge (i,j) splits the tree into:
      Subtree S = {nodes reachable from j}
      Rest    R = {all other nodes, including source}

    Replace (i,j) with any non-tree edge (u,v) bridging S and R.
    Both directions u->v and v->u are tested.
    First-improvement strategy: restart on first improvement found.

    Returns (improved_tree, improved_cost).
    """
    N = data["N"]

    improved = True
    while improved:
        improved = False
        for (i, j) in list(tree_edges):
            subtree = get_subtree((i, j), tree_edges, data)
            outside = set(range(N)) - subtree

            for u in list(subtree):
                for v in list(outside):
                    for edge in [(u, v), (v, u)]:
                        if not ((edge[0] in subtree and edge[1] in outside) or
                                (edge[0] in outside and edge[1] in subtree)):
                            continue
                        new_edges = [e for e in tree_edges if e != (i, j)]
                        new_edges.append(edge)
                        if not is_valid_tree(new_edges, data):
                            continue
                        nc = evaluate(new_edges, data)
                        if nc < current_cost - 1e-9:
                            tree_edges   = new_edges
                            current_cost = nc
                            improved     = True
                            break
                    if improved:
                        break
                if improved:
                    break
            if improved:
                break

    return tree_edges, current_cost


def move_reorientation(tree_edges: list, data: dict,
                        current_cost: float) -> tuple:
    """
    Reorientation MOVE (intensification operator):

    For each tree edge (i,j), try reversing direction to (j,i).
    Accept only if the reversed edge still yields a valid directed tree
    and the cost improves.
    First-improvement strategy.

    Returns (improved_tree, improved_cost).
    """
    improved = True
    while improved:
        improved = False
        for idx, (i, j) in enumerate(tree_edges):
            new_edges      = list(tree_edges)
            new_edges[idx] = (j, i)
            if not is_valid_tree(new_edges, data):
                continue
            nc = evaluate(new_edges, data)
            if nc < current_cost - 1e-9:
                tree_edges   = new_edges
                current_cost = nc
                improved     = True
                break

    return tree_edges, current_cost


def local_search(tree_edges: list, data: dict) -> tuple:
    """
    Apply 2opt-Swap and Reorientation alternately until no improvement.
    Returns the locally optimal tree and its cost.
    """
    cost = evaluate(tree_edges, data)

    improved = True
    while improved:
        improved = False

        new_edges, new_cost = move_2opt_swap(tree_edges, data, cost)
        if new_cost < cost - 1e-9:
            tree_edges = new_edges
            cost       = new_cost
            improved   = True

        new_edges, new_cost = move_reorientation(tree_edges, data, cost)
        if new_cost < cost - 1e-9:
            tree_edges = new_edges
            cost       = new_cost
            improved   = True

    return tree_edges, cost


# =============================================================================
# 7.  PERTURBATION  (diversification)
# =============================================================================

def perturbation(tree_edges: list, data: dict) -> list:
    """
    Double-Bridge-style perturbation (diversification operator):

    1. Choose a random tree edge (i,j) to remove.
    2. The tree splits into two components:
         Subtree S (rooted at j) and Rest R (containing source).
    3. Pick a random reconnecting edge (u,v) with one endpoint in S,
       the other in R, different from both (i,j) and (j,i).
    4. The new edge must produce a valid feasible tree.

    Up to 100 attempts; returns original tree unchanged if all fail.
    """
    N = data["N"]

    for _ in range(100):
        idx     = random.randrange(len(tree_edges))
        i, j    = tree_edges[idx]
        subtree = get_subtree((i, j), tree_edges, data)
        outside = set(range(N)) - subtree

        candidates = []
        for u in subtree:
            for v in outside:
                if (u, v) != (i, j) and (u, v) != (j, i):
                    candidates.append((u, v))
                if (v, u) != (i, j) and (v, u) != (j, i):
                    candidates.append((v, u))

        random.shuffle(candidates)

        for (u, v) in candidates[:20]:
            new_edges = [e for k, e in enumerate(tree_edges) if k != idx]
            new_edges.append((u, v))
            if is_valid_tree(new_edges, data) and \
               evaluate(new_edges, data) < float("inf"):
                return new_edges

    return list(tree_edges)   # fallback: unchanged


# =============================================================================
# 8.  ILS MAIN LOOP
# =============================================================================

def ils(data: dict,
        max_iterations: int = 300,
        max_no_improve: int = 60,
        seed: int = 42,
        verbose: bool = True) -> tuple:
    """
    Iterative Local Search main procedure.

    Pseudocode (from lecture03):
        s0     <- GreedyInitialSolution()
        s*     <- LocalSearch(s0)
        s_best <- s*
        while stopping criterion not met:
            s'   <- Perturbation(s*)        // diversification
            s*'  <- LocalSearch(s')         // intensification
            s*   <- AcceptanceCriterion(s*, s*')
            if cost(s*) < cost(s_best):
                s_best <- s*
        return s_best

    Stopping criterion: max_iterations reached OR no improvement for
    max_no_improve consecutive iterations.

    Parameters
    ----------
    data            : problem data dict from load_data()
    max_iterations  : hard iteration cap
    max_no_improve  : early-stop patience
    seed            : random seed for reproducibility
    verbose         : print progress

    Returns
    -------
    (best_tree_edges, best_cost, cost_history_per_iteration)
    """
    random.seed(seed)

    # Step 0 – Greedy initial solution
    s0          = greedy_initial_solution(data)
    greedy_cost = evaluate(s0, data)
    if verbose:
        print(f"  Greedy initial solution cost : {greedy_cost:.4f}")

    # Step 1 – First local search
    s_star, cost_star = local_search(s0, data)
    if verbose:
        print(f"  After initial local search   : {cost_star:.4f}")

    s_best     = list(s_star)
    cost_best  = cost_star
    history    = [cost_best]
    no_improve = 0

    # Main ILS loop
    for iteration in range(1, max_iterations + 1):

        # Perturbation (diversification)
        s_prime = perturbation(s_star, data)

        # Local search on perturbed solution (intensification)
        s_prime_star, cost_prime = local_search(s_prime, data)

        # Acceptance criterion: accept if improves current local optimum
        if cost_prime < cost_star - 1e-9:
            s_star    = s_prime_star
            cost_star = cost_prime

        # Update global best
        if cost_star < cost_best - 1e-9:
            s_best     = list(s_star)
            cost_best  = cost_star
            no_improve = 0
            if verbose:
                print(f"  Iter {iteration:4d} | NEW BEST : {cost_best:.4f}")
        else:
            no_improve += 1

        history.append(cost_best)

        if no_improve >= max_no_improve:
            if verbose:
                print(f"  Early stop at iter {iteration} "
                      f"({max_no_improve} iterations without improvement).")
            break

    return s_best, cost_best, history


# =============================================================================
# 9.  RESULT DISPLAY
# =============================================================================

def print_solution(tree_edges: list, cost: float, data: dict) -> None:
    """Print a detailed human-readable solution summary."""
    N      = data["N"]
    source = data["source"]
    l      = data["l"]
    D      = data["D"]
    lam    = data["lam"]
    c_rev  = data["c_rev"]
    c_om   = data["c_om"]
    c_fix  = data["c_fix"]
    c_var  = data["c_var"]
    alpha  = data["alpha"]
    p_umd  = data["p_umd"]
    c_heat = data["c_heat"]
    T_flh  = data["T_flh"]
    beta   = data["beta"]
    coords = data["coords"]

    result = compute_flows(tree_edges, data)
    if result is None:
        print("  !! INFEASIBLE SOLUTION !!")
        return

    P_in, total_source_out = result

    print("\n" + "=" * 65)
    print("  DISTRICT HEATING NETWORK  –  ILS OPTIMAL SOLUTION")
    print("=" * 65)
    print(f"  Objective (total annual expenses) : {cost:.4f} euro/a")
    print(f"  Source node (0-indexed)           : {source}")
    print(f"  Number of constructed edges       : {len(tree_edges)}  (= N-1)")
    print()
    print(f"  {'Edge':>6}  {'Length (m)':>10}  {'P_in (kW)':>12}  "
          f"{'P_out (kW)':>12}")
    print("  " + "-" * 48)

    constructed = set()
    for (i, j) in tree_edges:
        constructed.add((min(i, j), max(i, j)))
        p_out_val = (P_in[(i, j)] * (1 - data["theta_var"][i][j] * l[i][j])
                     - data["d"][i][j] * beta * lam
                     - data["theta_fix"][i][j] * l[i][j])
        print(f"  {i}->{j}  {l[i][j]:10.2f}  "
              f"{P_in[(i,j)]:12.4f}  {max(0.0, p_out_val):12.4f}")

    # Cost breakdown
    revenue     = sum(D[i][j] * lam * c_rev[i][j]           for (i,j) in tree_edges)
    heat_gen    = total_source_out * T_flh * c_heat[source] / beta
    maintenance = sum(l[i][j] * c_om[i][j]                  for (i,j) in tree_edges)
    fixed_inv   = alpha * sum(l[i][j] * c_fix                for (i,j) in tree_edges)
    var_inv     = alpha * sum(P_in[(i,j)] * l[i][j] * c_var[i][j] for (i,j) in tree_edges)
    unmet       = sum(D[i][j] * p_umd[i][j]
                      for i in range(N) for j in range(i+1, N)
                      if (i, j) not in constructed)

    print()
    print("  Annual cost breakdown (euro/a):")
    print(f"    Revenue              : -{revenue:15.4f}")
    print(f"    Heat generation      :  {heat_gen:15.4f}")
    print(f"    Maintenance          :  {maintenance:15.4f}")
    print(f"    Fixed investment     :  {fixed_inv:15.4f}")
    print(f"    Variable investment  :  {var_inv:15.4f}")
    print(f"    Unmet demand penalty :  {unmet:15.4f}")
    print("    " + "-" * 40)
    print(f"    TOTAL (= objective)  :  {cost:15.4f}")
    print("=" * 65)


def draw_network_ascii(tree_edges: list, data: dict) -> None:
    """Display the directed spanning tree in ASCII format."""
    source = data["source"]
    coords = data["coords"]

    children = defaultdict(list)
    for (i, j) in tree_edges:
        children[i].append(j)

    print("\n  Optimal network tree structure (rooted at source):")

    def print_subtree(node, prefix="", is_last=True):
        connector = "L-- " if is_last else "+-- "
        tag  = " <-- SOURCE" if node == source else ""
        x, y = coords[node]
        print(f"  {prefix}{connector}Node {node}{tag}  ({x:.0f}, {y:.0f})")
        ch = sorted(children[node])
        for k, child in enumerate(ch):
            ext = "    " if is_last else "|   "
            print_subtree(child, prefix + ext, k == len(ch) - 1)

    print_subtree(source, "", True)


# =============================================================================
# 10.  ENTRY POINT
# =============================================================================

def find_data_file(name: str) -> str:
    """Search common locations for the data file."""
    candidates = [
        name,
        os.path.join("/mnt/user-data/uploads", name),
        os.path.join(os.path.dirname(os.path.abspath(__file__)), name),
    ]
    for path in candidates:
        if os.path.exists(path):
            return path
    raise FileNotFoundError(f"Data file not found: '{name}'")


if __name__ == "__main__":

    # ── Small instance ────────────────────────────────────────────────────────
    filename = find_data_file("InputDataEnergySmallInstance.json")
    print(f"\nLoading data from: {filename}")
    data = load_data(filename)
    print(f"Instance : {data['N']} nodes  |  source = node {data['source']}")

    print("\n-- ILS Metaheuristic (small instance) " + "-" * 30)
    t0 = time.time()

    best_tree, best_cost, history = ils(
        data,
        max_iterations=300,
        max_no_improve=60,
        seed=42,
        verbose=True,
    )

    elapsed = time.time() - t0
    print(f"\n  Total CPU time : {elapsed:.2f} s")
    print(f"  ILS iterations : {len(history) - 1}")

    print_solution(best_tree, best_cost, data)
    draw_network_ascii(best_tree, data)

    print(f"\nOptimal solutions")
    print(f"Small instance")
    print(f"The optimal value for the small instance is {best_cost:.8f}.")