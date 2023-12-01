import numpy as np
import matplotlib.pyplot as plt

# %% Truss structure data
E = 1e4  # Young modulus (ksi)
p = 0.1
s_lim = 25
d_lim = 2

nodes = []
bars = []

nodes.append([720, 360])
nodes.append([720, 0])
nodes.append([360, 360])
nodes.append([360, 0])
nodes.append([0, 360])
nodes.append([0, 0])

bars.append([4, 2])
bars.append([2, 0])
bars.append([5, 3])
bars.append([3, 1])
bars.append([3, 2])
bars.append([1, 0])
bars.append([4, 3])
bars.append([5, 2])
bars.append([2, 1])
bars.append([3, 0])

nodes = np.array(nodes).astype(float)
bars = np.array(bars)

# Apply Force
P = np.zeros_like(nodes)  # kips
P[1, 1] = -100
P[3, 1] = -100

# Support Displacement
Ur = [0, 0, 0, 0]

# Condition of degree of freedom (1 = free, 0 = fixed)
DOFCON = np.ones_like(nodes).astype(int)
DOFCON[4, :] = 0
DOFCON[5, :] = 0


# %% Truss structure analysis
def TrussAnalysis(A):
    NN = len(nodes)  # Number of nodes
    NE = len(bars)  # Number of bars
    DOF = 2  # 2D Truss
    NDOF = DOF * NN  # Total number of degree of freedom

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
    return S, Mass, U


# %% PROBLEM DEFINITION
d = 10
xMin, xMax = 0.1, 40
vMin, vMax = -0.2 * (xMax - xMin), 0.2 * (xMax - xMin)
MaxIt = 500
ps = 30
c1 = 2
c2 = 2
w = 0.9 - ((0.9 - 0.4) / MaxIt) * np.linspace(0, MaxIt, MaxIt)


def limitV(V):
    for i in range(len(V)):
        if V[i] > vMax:
            V[i] = vMax
        if V[i] < vMin:
            V[i] = vMin
    return V


def limitX(X):
    for i in range(len(X)):
        if X[i] > xMax:
            X[i] = xMax
        if X[i] < xMin:
            X[i] = xMin
    return X


# %% Algorithm
def Optimization():
    class Particle:
        def __init__(self):
            self.position = np.random.uniform(20, xMax, [ps, d])
            self.velocity = np.random.uniform(vMin, vMax, [ps, d])
            self.cost = np.zeros(ps)
            self.stress = np.zeros([ps, d])
            self.displacement = np.zeros([ps, len(nodes), 2])
            for i in range(ps):
                self.stress[i], self.cost[i], self.displacement[i] = TrussAnalysis(self.position[i])
            self.pbest = np.copy(self.position)
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
                                        + c2 * np.random.rand(d) * (self.gbest - self.position[i]))
                    self.velocity[i] = limitV(self.velocity[i])
                    self.position[i] = self.position[i] + self.velocity[i]
                    self.position[i] = limitX(self.position[i])
                    self.stress[i], self.cost[i], self.displacement[i] = TrussAnalysis(self.position[i])
                    C_total = 0
                    for cd in range(d):
                        if np.abs(self.stress[i, cd]) > s_lim:
                            C1 = np.abs((self.stress[i, cd] - s_lim) / s_lim)
                        else:
                            C1 = 0
                        C_total = C_total + C1
                    for cx in range(len(nodes)):
                        if np.abs(self.displacement[i, cx, 0]) > d_lim:
                            C2 = np.abs((self.displacement[i, cx, 0] - d_lim) / d_lim)
                        else:
                            C2 = 0
                        C_total = C_total + C2
                    for cy in range(len(nodes)):
                        if np.abs(self.displacement[i, cy, 1]) > d_lim:
                            C3 = np.abs((self.displacement[i, cy, 1] - d_lim) / d_lim)
                        else:
                            C3 = 0
                        C_total = C_total + C3
                    phi = (1 + C_total)
                    self.cost[i] = self.cost[i] * phi
                    if self.cost[i] < self.pbest_cost[i]:
                        self.pbest[i] = self.position[i]
                        self.pbest_cost[i] = self.cost[i]
                        if self.pbest_cost[i] < self.gbest_cost:
                            self.gbest = self.pbest[i]
                            self.gbest_cost = self.pbest_cost[i]
                self.BestCost[it] = self.gbest_cost
                self.BestPosition[it] = self.gbest

        def Plot(self):
            np.set_printoptions(precision=3, suppress=True)
            plt.plot(self.BestCost)
            print('Design Variables A [in2]')
            print(self.BestPosition[-1][np.newaxis].T)
            Stress, Cost, Disp = TrussAnalysis(self.BestPosition[-1])
            print('Stress [ksi]')
            print(Stress[np.newaxis].T)
            print('Displacement [in]')
            print(Disp)
            print('Best fitness value =', self.gbest_cost)

    a = Particle()
    a.Evaluate()
    a.Plot()


# %%Run
Optimization()
plt.show()