

class Estimator:
    def __init__(self, model: LatentSpaceModel, method="MLE"):
        self.model = model
        self.method = method

    def fit(self, Y, init_state: ParameterState):
        if self.method == "MLE":
            return self._fit_mle(Y, init_state)
        elif self.method == "MAP":
            return self._fit_map(Y, init_state)
        elif self.method == "MCMC":
            return self._fit_mcmc(Y, init_state)
        elif self.method == "VI":
            return self._fit_vi(Y, init_state)


