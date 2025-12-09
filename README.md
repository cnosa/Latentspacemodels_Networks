# LATENTSPACES_NETWORK_MANIFOLD

This repository contains experiments and analyses of latent space models applied to network data, using both Euclidean and spherical geometries. The main goal is to assess how well these models capture social structure across various historical and synthetic networks. The code was executed on a laptop with an Intel Core i5 processor, 24 GB of RAM, and a 512 GB SSD.

## Repository structure

- **`LATENTSPACES_NETWORK_MANIFOLD/`**
  - `AdditionalContent/` # Extra resources (images, tests)
  - `Example_Karate/` # Karate club example data & notebook
  - `Example_Monks/` # Monks network example & data
  - `Example_FlorentineFamilies/` # Florentine families dataset & analysis
    - `UsingHMC/` # HMC implementation, results & notebooks
  - `LSMN_P.py` # Main Python script (MH estimation)
  - `README.md`
  - `requirements.txt`

## Models

We implement latent space models in the following geometries:

- **$\mathbb{R}^1, \mathbb{R}^2, \mathbb{R}^3$**: Euclidean latent spaces
- **$\mathbb{S}^1, \mathbb{S}^2$**: Spherical latent spaces

Each model infers a posterior distribution over node positions, which is then used to interpret structural features of the observed network.

## Case studies

- **FlorentineFamilies**: Marriage network among 15th-century Florentine families, as analyzed by Padgett & Ansell (1993).
- **Karate Club**: Zachary’s karate club network (1977), used as a benchmark example.
- **Monks**: Placeholder for future experiments with the monk social network dataset.

## Requirements

This project requires the following Python (3.11.7) libraries:

- [networkx](https://networkx.org/) — Creation and manipulation of graphs.
- [numpy](https://numpy.org/) — Numerical operations and random data generation.
- [matplotlib](https://matplotlib.org/) — Static data visualization.
- [pandas](https://pandas.pydata.org/) — Data manipulation and analysis using DataFrames.
- [seaborn](https://seaborn.pydata.org/) — Statistical data visualization.
- [scipy](https://scipy.org/) — Scientific functions and tools, including `scipy.stats` and `scipy.special`.
- [plotly](https://plotly.com/python/) — Interactive visualizations (`plotly.express`, `plotly.graph_objects`).
- [tqdm](https://tqdm.github.io/) — Progress bars for iterations.
- [ridgeplot](https://pypi.org/project/ridgeplot/) — Overlaid density plots.

You can install all dependencies with:

```bash
pip install -r requirements.txt
```

## Citation

If you use this repository or its results in your research, please cite it as:

```bibtex
@article{sosanosa2025spherical,
  title   = {Spherical latent space models for social network analysis},
  author  = {Sosa, Juan and Nosa, Carlos},
  journal = {Universidad Nacional de Colombia},
  year    = {2025},
  abstract= {This article introduces a spherical latent space model for social network analysis,
              embedding actors on a hypersphere rather than in Euclidean space as in standard
              latent space models. The spherical geometry facilitates the representation of transitive
              relationships and community structure, naturally captures cyclical patterns, and ensures
              bounded distances, thereby mitigating degeneracy issues common in traditional approaches.
              Bayesian inference is performed via Markov chain Monte Carlo methods to estimate both latent
              positions and other model parameters. The approach is demonstrated using two benchmark social
              network datasets, yielding improved model fit and interpretability relative to conventional
              latent space models.},
  keywords = {Bayesian inference, latent space models, network analysis, spherical geometry, social networks}
}
```