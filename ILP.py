import pulp
import numpy as np
import openpyxl
import math

wb = openpyxl.load_workbook('InputDataEnergySmallInstance.xlsx')

def read_matrix(sheet_name, n=8):
    ws = wb[sheet_name]
    return [[float(ws.cell(i+1, j+1).value or 0) for j in range(n)] for i in range(n)]

def read_vector(sheet_name, n=8):
    ws = wb[sheet_name]
    return [float(ws.cell(i+1, 1).value or 0) for i in range(n)]

def read_scalar(sheet_name):
    return float(wb[sheet_name].cell(1,1).value)

N = 8
source = int(read_scalar('SourceNum')) - 1  # 0-indexed -> 3

coords = [(float(wb['NodesCord'].cell(i+1,1).value), float(wb['NodesCord'].cell(i+1,2).value)) for i in range(N)]
l = [[math.sqrt((coords[i][0]-coords[j][0])**2 + (coords[i][1]-coords[j][1])**2) for j in range(N)] for i in range(N)]

theta_fix = read_matrix('vfix(thetaijfix)')
theta_var = read_matrix('vvar(thetaijvar)')
c_fix = read_scalar('FixedUnitCost')
c_var = read_matrix('cvar(cijvar)')
c_heat = read_vector('cheat(ciheat)')
c_om = read_matrix('com(cijom)')
c_rev = read_matrix('crev(cijrev)')
T_flh = read_scalar('Tflh(Tiflh)')
beta = read_scalar('Betta')
lam = read_scalar('Lambda')
alpha = float(wb['Alpha'].cell(1,1).value)
d = read_matrix('EdgesDemandPeak(dij)')
D = read_matrix('EdgesDemandAnnual(Dij)')
C_max = read_matrix('Cmax(cijmax)')
Q_max = read_scalar('SourceMaxCap(Qimax)')
p_umd = read_matrix('pumd(pijumd)')

nodes = list(range(N))
edges = [(i, j) for i in nodes for j in nodes if i != j]

prob = pulp.LpProblem("DistrictHeating", pulp.LpMinimize)

X = {(i,j): pulp.LpVariable(f"X_{i}_{j}", cat='Binary') for (i,j) in edges}
P_in = {(i,j): pulp.LpVariable(f"Pin_{i}_{j}", lowBound=0) for (i,j) in edges}
P_out = {(i,j): pulp.LpVariable(f"Pout_{i}_{j}", lowBound=0) for (i,j) in edges}

revenue = pulp.lpSum(X[i,j] * D[i][j] * lam * c_rev[i][j] for (i,j) in edges)
heat_gen = pulp.lpSum(P_in[source,j] for j in nodes if j != source) * T_flh * c_heat[source] / beta
maintenance = pulp.lpSum(X[i,j] * l[i][j] * c_om[i][j] for (i,j) in edges)
fixed_inv = alpha * pulp.lpSum(X[i,j] * l[i][j] * c_fix for (i,j) in edges)
var_inv = alpha * pulp.lpSum(P_in[i,j] * l[i][j] * c_var[i][j] for (i,j) in edges)
unmet = pulp.lpSum((1 - X[i,j] - X[j,i]) * D[i][j] * p_umd[i][j]
                   for i in nodes for j in nodes if i < j)

prob += heat_gen + maintenance + fixed_inv + var_inv + unmet - revenue

# C1: tree structure
prob += pulp.lpSum(X[i,j] for (i,j) in edges) == N - 1

# C2: unidirectionality
for i in nodes:
    for j in nodes:
        if i < j:
            prob += X[i,j] + X[j,i] <= 1

# C3: demand satisfaction
for (i,j) in edges:
    delta = d[i][j] + theta_fix[i][j] * l[i][j]
    prob += P_out[i,j] == P_in[i,j] * (1 - theta_var[i][j] * l[i][j]) - delta * X[i,j]

# C4: flow equilibrium at non-source nodes
for j in nodes:
    if j != source:
        prob += pulp.lpSum(P_out[i,j] for i in nodes if i != j) == \
               pulp.lpSum(P_in[j,i] for i in nodes if i != j)

# C5: edge capacity
for (i,j) in edges:
    prob += P_in[i,j] <= C_max[i][j] * X[i,j]

# C6: no edges into source
for j in nodes:
    if j != source:
        prob += X[j, source] == 0

# C7: source generation capacity
prob += pulp.lpSum(P_in[source,j] for j in nodes if j != source) <= Q_max

# C8: tour elimination
for j in nodes:
    if j != source:
        prob += pulp.lpSum(X[i,j] for i in nodes if i != j) >= 1

prob.solve(pulp.PULP_CBC_CMD(msg=1))

print(f"\nStatus: {pulp.LpStatus[prob.status]}")
print(f"Objective (Total Expenses): {pulp.value(prob.objective):.2f} €/a")

print("\nSelected edges:")
for (i,j) in edges:
    if pulp.value(X[i,j]) and pulp.value(X[i,j]) > 0.5:
        print(f"  {i} -> {j}  |  Pin={pulp.value(P_in[i,j]):.2f} kW  Pout={pulp.value(P_out[i,j]):.2f} kW")