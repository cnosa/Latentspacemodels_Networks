#Cómo implementar mi modelo en POO? 
# Estaba pensando en tres clases grandes: una que genere modelos, otra que haga la estimación y otra que haga el análisis luego de la estimación 
# También necesito un método de simulación de redes con un número de nodos determinado por el usuario. 
# Lo pongo en la clase que genere modelos? 
# En la simulación, estimación y análisis debo ser capaz de fijar unos parámetros para no simular, estimar o analizar 
# La estimación tiene que ser pensada por optimización (MLE o MAP o variacional), bayesiana (muestreo) 
# También quiero que al futuro esto se pueda usar como base para un modelo multicapa o temporal 
# Solo quiero 3 clases, será adecuado para una librería?


class LatentSpaceModel:
    def __init__(self, kappa, dim, link="logit"):
        self.kappa = kappa
        self.dim = dim
        self.link = link

    def distance(self, z_i, z_j):
        ...

    def predictor(self, z_i, z_j, eta_i, eta_j, alpha):
        ...

    def log_likelihood(self, Y, Z, eta, alpha):
        ...

    def log_prior(self, Z, eta, alpha):
        return 0.0  # por defecto (frecuentista)

    def simulate(self, n_nodes, params, seed=None):
        """
        Genera (Z, eta, Y)
        """


class ParameterState:
    def __init__(self, Z=None, eta=None, alpha=None, kappa=None):
        self.Z = Z
        self.eta = eta
        self.alpha = alpha
        self.kappa = kappa

    def is_fixed(self, name):
        return getattr(self, name) is not None
