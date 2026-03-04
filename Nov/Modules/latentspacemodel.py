import re

class FormulaParser:
    
    def __init__(self, formula):
        self.formula = formula
        self.node_terms = []
        self.edge_terms = []
        self.use_distance = False
        self.use_eta_deformation = False
        self.include_intercept = False
        self._parse()

    def _parse(self):
        
        rhs = self.formula.split("~")[1]
        terms = [t.strip() for t in rhs.split("+")]

        for t in terms:

            if t == "1":
                self.include_intercept = True

            elif t.startswith("node("):
                var = re.findall(r"node\((.*?)\)", t)[0]
                self.node_terms.append(var)

            elif t.startswith("edge("):
                var = re.findall(r"edge\((.*?)\)", t)[0]
                self.edge_terms.append(var)

            elif t == "dist":
                self.use_distance = True

            elif t == "dist_eta":
                self.use_distance = True
                self.use_eta_deformation = True

import numpy as np

class ParameterState:

    def __init__(self, Z=None, eta=None, beta=None, kappa=None, fixed=None):

        self.Z = Z
        self.eta = eta
        self.beta = beta
        self.kappa = kappa
        self.fixed = fixed or {}

    def is_fixed(self, name):
        return self.fixed.get(name, False)

    def set_fixed(self, name, value=True):
        self.fixed[name] = value

    def free_vector(self):

        parts = []

        if not self.is_fixed("Z") and self.Z is not None:
            parts.append(self.Z.ravel())

        if not self.is_fixed("eta") and self.eta is not None:
            parts.append(self.eta.ravel())

        if not self.is_fixed("beta") and self.beta is not None:
            parts.append(np.atleast_1d(self.beta))

        if not parts:
            return np.array([])

        return np.concatenate(parts)

    def update_from_vector(self, vec, template):

        idx = 0

        if not self.is_fixed("Z") and template.Z is not None:
            size = template.Z.size
            self.Z = vec[idx:idx+size].reshape(template.Z.shape)
            idx += size

        if not self.is_fixed("eta") and template.eta is not None:
            size = template.eta.size
            self.eta = vec[idx:idx+size]
            idx += size

        if not self.is_fixed("beta") and template.beta is not None:
            size = np.atleast_1d(template.beta).size
            self.beta = vec[idx:idx+size]
            idx += size



import numpy as np
from scipy.special import expit


class LatentSpaceModel:

    def __init__(self, formula, kappa=0, dim=2, link="logit"):

        self.kappa = kappa
        self.dim = dim
        self.link = link
        self.formula = FormulaParser(formula)

    # -------------------------------------------------
    # DISTANCIA
    # -------------------------------------------------
    def distance(self, z_i, z_j):

        if self.kappa == 0:
            return np.linalg.norm(z_i - z_j)

        elif self.kappa == 1:
            return np.arccos(np.clip(np.dot(z_i, z_j), -1.0, 1.0))

        elif self.kappa == -1:
            inner = z_i[0]*z_j[0] + z_i[1]*z_j[1] - z_i[2]*z_j[2]
            return np.arccosh(-inner)

        else:
            raise ValueError("kappa debe ser -1, 0 o 1")

    # -------------------------------------------------
    # PREDICTOR GENERAL
    # -------------------------------------------------
    def predictor(self, i, j, state, X_node=None, X_edge=None):

        s = 0.0
        beta = state.beta
        idx = 0

        # Intercepto
        if self.formula.include_intercept:
            s += beta[idx]
            idx += 1

        # Efectos nodales
        for var in self.formula.node_terms:

            x_i = X_node[var][i]
            x_j = X_node[var][j]

            s += beta[idx] * x_i
            idx += 1

            s += beta[idx] * x_j
            idx += 1

        # Efectos relacionales
        for var in self.formula.edge_terms:

            x_ij = X_edge[var][i, j]
            s += beta[idx] * x_ij
            idx += 1

        # ---------------------------
        # DISTANCIA
        # ---------------------------
        if self.formula.use_distance:

            z_i = state.Z[i]
            z_j = state.Z[j]

            d = self.distance(z_i, z_j)

            if self.formula.use_eta_deformation:

                mult = (np.exp(state.eta[i]) + np.exp(state.eta[j])) / 2
                s -= mult * d

            else:
                s -= d

        return s
    # -------------------------------------------------
    # LOG-LIKELIHOOD
    # -------------------------------------------------
    def log_likelihood(self, Y, state, X_node=None, X_edge=None):

        Z = state.Z
        eta = state.eta
        beta = state.beta
        n = Y.shape[0]

        # ----------------------------------------
        # Construcción del predictor S (matriz)
        # ----------------------------------------
        S = np.zeros((n, n))
        idx = 0

        # Intercepto
        if self.formula.include_intercept:
            S += beta[idx]
            idx += 1

        # Efectos nodales
        for var in self.formula.node_terms:

            x = X_node[var]

            S += beta[idx] * x[:, None]
            idx += 1

            S += beta[idx] * x[None, :]
            idx += 1

        # Efectos relacionales
        for var in self.formula.edge_terms:
            S += beta[idx] * X_edge[var]
            idx += 1

        # ----------------------------------------
        # Distancia Euclidiana vectorizada
        # ----------------------------------------
        if self.formula.use_distance:

            if self.kappa != 0:
                raise NotImplementedError("Vectorización solo implementada para κ=0")

            # ||z_i - z_j||^2 = ||z_i||^2 + ||z_j||^2 - 2 z_i^T z_j
            norm_sq = np.sum(Z**2, axis=1)
            D2 = norm_sq[:, None] + norm_sq[None, :] - 2 * Z @ Z.T
            D2 = np.maximum(D2, 0.0)
            D = np.sqrt(D2)

            if self.formula.use_eta_deformation:
                mult = (np.exp(eta)[:, None] + np.exp(eta)[None, :]) / 2
                S -= mult * D
            else:
                S -= D

        # ----------------------------------------
        # Usar solo triángulo superior
        # ----------------------------------------
        iu = np.triu_indices(n, k=1)

        s = S[iu]
        y = Y[iu]

        # log-likelihood estable
        ll = np.sum(y * s - np.logaddexp(0, s))

        return ll

    # -------------------------------------------------
    # SIMULACIÓN CORRECTA CON FÓRMULA
    # -------------------------------------------------
    def simulate(self, n_nodes, state=None,
                 X_node=None,
                 X_edge=None,
                 seed=None):

        rng = np.random.default_rng(seed)

        # -------------------------------------------------
        # Generar Z y eta si no vienen en state
        # -------------------------------------------------
        if state is None:
            Z = rng.normal(size=(n_nodes, self.dim))
            eta = rng.normal(size=n_nodes)

            # tamaño de beta automático
            p = 0

            if self.formula.include_intercept:
                p += 1

            p += 2 * len(self.formula.node_terms)
            p += len(self.formula.edge_terms)

            # distancia con coeficiente
            if self.formula.dist_type in ["plain", "eta"]:
                p += 1
            beta = rng.normal(size=p)

            state = ParameterState(
                Z=Z,
                eta=eta,
                beta=beta,
                kappa=self.kappa
            )

        # -------------------------------------------------
        # Generar X_node si no existe
        # -------------------------------------------------
        if X_node is None:
            X_node = {}
            for var in self.formula.node_terms:
                X_node[var] = rng.normal(size=n_nodes)

        # -------------------------------------------------
        # Generar X_edge si no existe
        # -------------------------------------------------
        if X_edge is None:
            X_edge = {}
            for var in self.formula.edge_terms:
                mat = rng.normal(size=(n_nodes, n_nodes))
                mat = (mat + mat.T) / 2
                np.fill_diagonal(mat, 0.0)
                X_edge[var] = mat

        # -------------------------------------------------
        # Generar matriz Y
        # -------------------------------------------------
        Y = np.zeros((n_nodes, n_nodes))

        for i in range(n_nodes):
            for j in range(i+1, n_nodes):

                s = self.predictor(i, j, state, X_node, X_edge)
                p_ij = expit(s)

                y = rng.binomial(1, p_ij)

                Y[i, j] = y
                Y[j, i] = y

        return state, Y, X_node, X_edge
