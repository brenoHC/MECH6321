import numpy as np
import matplotlib.pyplot as plt

# Truss properties
E = 1e4       # Young's modulus
p = 0.1111    # Material Density
s_lim = 25    # Stress Limit (ksi)
# d_lim = 2   # Displacement Limit

# array node & bar locations from 0 to 7 by 0 to 360 respectability
nodes = []
bars = []

nodes.append([0, 120])
nodes.append([120, 120])
nodes.append([240, 120])
nodes.append([360, 120])
nodes.append([0, 0])
nodes.append([120, 0])
nodes.append([240, 0])
nodes.append([360, 0])

bars.append([0, 1])
bars.append([1, 2])
bars.append([2, 3])
bars.append([4, 5])
bars.append([5, 6])
bars.append([6, 7])

bars.append([5, 1])
bars.append([6, 2])
bars.append([7, 3])

bars.append([0, 5])
bars.append([4, 1])
bars.append([1, 6])
bars.append([5, 2])
bars.append([2, 7])
bars.append([6, 3])

nodes = np.array(nodes).astype(float)
bars = np.array(bars)

nodes_original = np.copy(nodes)

# Force applied in  (kips)
P = np.zeros_like(nodes)
P[7, 1] = -10

# Support Displacement
Ur = [0, 0, 0, 0]

# Condition of degree of freedom (1 = free, 0 = fixed)
DOFCON = np.ones_like(nodes).astype(int)
DOFCON[0, :] = 0
DOFCON[4, :] = 0

# Permissible area and geometry variables
Ai = [0.111, 0.141, 0.174, 0.220, 0.270, 0.287, 0.347, 0.440, 0.539, 0.954,
      1.081, 1.174, 1.333, 1.488, 1.764, 2.142, 2.697, 2.800, 3.131, 3.565,
      3.813, 4.805, 5.952, 6.572, 7.192, 8.525, 9.300, 10.850, 13.330, 14.290,
      17.170, 19.180]  # Return evenly spaced numbers over a specified interval
A_index = np.linspace(0, len(Ai) - 1, len(Ai)).astype(int)

NC = []  # Node Conditions
NC.append([100, 140])  # X2 [min, max]
NC.append([220, 260])  # X3 [min, max]
NC.append([100, 140])  # Y2 [min, max]
NC.append([100, 140])  # Y3 [min, max]
NC.append([50, 90])    # Y4 [min, max]
NC.append([-20, 20])   # Y6 [min, max]
NC.append([-20, 20])   # Y7 [min, max]
NC.append([20, 60])    # Y8 [min, max]
NC = np.array(NC)


# Assign geometry variable
def AssignGeometry(Var, d, d1): # seeting the coordinates of the node, it will change at every iteration
    NCi = Var[d1:d]  # Node Conditions

    # Assign geometry variable
    nodes[1, 0] = NCi[0]  # Node 2 (Axis X)
    nodes[5, 0] = NCi[0]  # Node 6 (Axis X)
    nodes[2, 0] = NCi[1]  # Node 3 (Axis X)
    nodes[6, 0] = NCi[1]  # Node 7 (Axis X)
    nodes[1, 1] = NCi[2]  # Node 2 (Axis Y)
    nodes[2, 1] = NCi[3]  # Node 3 (Axis Y)
    nodes[3, 1] = NCi[4]  # Node 4 (Axis Y)
    nodes[5, 1] = NCi[5]  # Node 6 (Axis Y)
    nodes[6, 1] = NCi[6]  # Node 7 (Axis Y)
    nodes[7, 1] = NCi[7]  # Node 8 (Axis Y)


# %% Truss structure analysis
def TrussAnalysis(Var, d, d1):
    NN = len(nodes)  # Number of nodes
    NE = len(bars)   # Number of bars
    DOF = 2          # 2D Truss
    NDOF = DOF * NN  # Total # of DoF

    A = np.copy(Var[0:d1])
    for i in range(NE):
        A[i] = Ai[Var[i].astype(int)]
    AssignGeometry(Var, d, d1)

    # Structural analysis
    d = nodes[bars[:, 1], :] - nodes[bars[:, 0], :]
    L = np.sqrt((d ** 2).sum(axis=1))
    angle = d.T / L
    a = np.concatenate((-angle.T, angle.T), axis=1)
    K = np.zeros([NDOF, NDOF])  # Stiffness matrix
    for k in range(NE):
        # DOFs
        aux = 2 * bars[k, :]
        index = np.r_[aux[0]:aux[0] + 2, aux[1]:aux[1] + 2]
        # Stiffness matrix
        ES = np.dot(a[k][np.newaxis].T * E * A[k], a[k][np.newaxis]) / L[k]
        K[np.ix_(index, index)] = K[np.ix_(index, index)] + ES
    freeDOF = DOFCON.flatten().nonzero()[0]
    supportDOF = (DOFCON.flatten() == 0).nonzero()[0]
    Kff = K[np.ix_(freeDOF, freeDOF)]
    Pf = P.flatten()[freeDOF]
    Uf = np.linalg.solve(Kff, Pf)
    U = DOFCON.astype(float).flatten()
    U[freeDOF] = Uf
    U[supportDOF] = Ur
    U = U.reshape(NN, DOF)  # Displacement (in)
    u = np.concatenate((U[bars[:, 0]], U[bars[:, 1]]), axis=1)
    N = E * A[:] / L[:] * (a[:] * u[:]).sum(axis=1)  # Force (kips)
    S = N / A
    Mass = (p * A * L).sum()
    return S, Mass


#  PROBLEM DEFINITION
d1 = len(bars) #
d2 = len(NC)
d = d1 + d2

# Lower and upper bound of search area
Xlim = [min(A_index), max(A_index)] * np.ones([len(bars), 2])
Xlim = np.concatenate([Xlim, NC], axis=0)

# Lower and upper bound of velocity
Vlim = np.zeros([d, 2])
Vlim[:, 1] = (Xlim[:, 1] - Xlim[:, 0]) * 0.2
Vlim[:, 0] = -Vlim[:, 1]

MaxIt = 500
ps = 9  # according to paper Shi and aberhart
c1 = 2  # according to paper Shi and aberhart
c2 = 2  # according to paper Shi and aberhart
w = 0.9 - ((0.9 - 0.4) / MaxIt) * np.linspace(0, MaxIt, MaxIt)


def limitV(V, d):  # defining limit of velocity
    for j in range(d):
        if V[j] > Vlim[j, 1]:
            V[j] = Vlim[j, 1]
        if V[j] < Vlim[j, 0]:
            V[j] = Vlim[j, 0]
    return V


def Plot(nodes, X, D, D1, Assign, c, lt, lw, lg):
    if Assign == 1:
        AssignGeometry(X, D, D1)
    for i in range(D1):
        xi, xf = nodes[bars[i, 0], 0], nodes[bars[i, 1], 0]
        yi, yf = nodes[bars[i, 0], 1], nodes[bars[i, 1], 1]
        line, = plt.plot([xi, xf], [yi, yf], color=c, linestyle=lt, linewidth=lw)
    line.set_label(lg)
    plt.legend(prop={'size': 8})



def FindNearest(array, value): # finds integer
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx]


#  PSO Algorithm
def optimization():
    class Particle:
        def __init__(self):
            self.position = np.zeros([ps, d])
            self.velocity = np.zeros([ps, d])
            self.cost = np.zeros(ps)
            self.stress = np.zeros([ps, d1])
            for i in range(ps):
                for j in range(d):
                    if j < d1:
                        self.position[i, j] = np.random.choice(A_index[A_index > 15]) # picks random value selected > 15
                    else:
                        self.position[i, j] = np.random.uniform(Xlim[j, 0], Xlim[j, 1])
                    self.velocity[i, j] = np.random.uniform(Vlim[j, 0], Vlim[j, 1])
                self.stress[i], self.cost[i] = TrussAnalysis(self.position[i], d, d1)
            self.pbest = np.copy(self.position) # perosnal best for all varibles (used for initial calculations only)
            self.pbest_cost = np.copy(self.cost)
            self.index = np.argmin(self.pbest_cost)
            self.gbest = self.pbest[self.index]
            self.gbest_cost = self.pbest_cost[self.index]
            self.BestCost = np.zeros(MaxIt)
            self.BestPosition = np.zeros([MaxIt, d])

        def Evaluate(self):
            for it in range(MaxIt):
                for i in range(ps):
                    self.velocity[i] = (w[it] * self.velocity[i]
                                        + c1 * np.random.rand(d) * (self.pbest[i] - self.position[i])
                                        + c2 * np.random.rand(d) * (self.gbest - self.position[i])) # Velocity equation from paper
                    self.velocity[i] = limitV(self.velocity[i], d)
                    self.position[i] = self.position[i] + self.velocity[i] # update position
                    for p in range(d1):
                        self.position[i, p] = FindNearest(A_index, self.position[i, p])
                    self.stress[i], self.cost[i] = TrussAnalysis(self.position[i], d, d1)

                    C_total = 0 # Desing constrain(5) from farqad Jawad et al. controls stress from opt
                    for cd in range(d1):
                        if np.abs(self.stress[i, cd]) > s_lim:
                            C1 = np.abs((self.stress[i, cd] - s_lim) / s_lim)
                        else:
                            C1 = 0
                        C_total = C_total + C1
                    phi = (1 + C_total)
                    self.cost[i] = self.cost[i] * phi
                    if self.cost[i] < self.pbest_cost[i]:
                        self.pbest[i] = self.position[i]
                        self.pbest_cost[i] = self.cost[i]
                        if self.pbest_cost[i] < self.gbest_cost:
                            self.gbest = self.pbest[i]
                            self.gbest_cost = self.pbest_cost[i]
                self.BestCost[it] = self.gbest_cost # collects data
                self.BestPosition[it] = self.gbest

        def Plot(self):
            np.set_printoptions(precision=3, suppress=True)
            plt.figure(0)
            plt.plot(self.BestCost)
            print('Design Variables')
            Design_var = np.copy(self.position[-1])
            for i in range(d1):
                Design_var[i] = Ai[self.position[-1, i].astype(int)]
            print(Design_var[np.newaxis].T)
            Stress, Cost = TrussAnalysis(self.BestPosition[-1], d, d1)
            print('Stress [ksi]')
            print(Stress[np.newaxis].T) # T = transpose
            print('Best fitness value COST function (weight)=', self.gbest_cost)
            plt.figure(1)
            Plot(nodes_original, Design_var, d, d1, 0, 'gray', '--', 1, 'Original design')
            Plot(nodes, Design_var, d, d1, 1, 'red', '-', 1, 'Optimal design')


    a = Particle()
    a.Evaluate()
    a.Plot()



#  Run optimization
optimization()
plt.show()