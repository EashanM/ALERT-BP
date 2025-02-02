import numpy as np
import pandas as pd
import torch
import pyro
import pyro.distributions as dist
from pyro.infer import MCMC, NUTS
import arviz as az

df = pd.read_csv("ALERTBP_Data_cleaned.csv")

fixed_effects = [
    'gestage', 'csex', 'msmkhist', 'priordiabp', 'ppbmi', 'everbfed',
    'gdm_report', 'agemom', 'mblncig', 'mblhdis', 'ppwgt', 'mblcvd',
    'mblsmkexp', 'cbthweight', 'gestwgtgain', 'stopbfed', 'Nethnic_mom'
]

random_slope_vars = ['age', 'height', 'bmi']

y = torch.tensor(df["SBP"].values, dtype=torch.float32)
X_fixed = torch.tensor(df[fixed_effects].values, dtype=torch.float32)
Z_random = torch.tensor(df[random_slope_vars].values, dtype=torch.float32)

ids = df["id"].values
unique_ids = np.unique(ids)
id_map = {u: i for i, u in enumerate(unique_ids)}
id_idx = torch.tensor([id_map[i] for i in ids], dtype=torch.long)

N = y.shape[0]
J = len(unique_ids)
K = len(fixed_effects)
R = len(random_slope_vars)

def model():
    intercept = pyro.sample("intercept", dist.Normal(0., 10.))
    betas = pyro.sample("betas", dist.Normal(torch.zeros(K), 10.*torch.ones(K)))

    eta = 2.0
    L = pyro.sample("L", dist.LKJCholesky(dim=R, concentration=torch.tensor(eta)))
    tau = pyro.sample("tau", dist.HalfNormal(10.*torch.ones(R)))

    z_b = pyro.sample("z_b", dist.Normal(torch.zeros(J, R), torch.ones(J, R)))
    z_b_scaled = z_b * tau
    b = torch.matmul(z_b_scaled, L)

    mu = intercept + (X_fixed * betas).sum(dim=1)
    random_contrib = (Z_random * b[id_idx]).sum(dim=1)
    mu = mu + random_contrib

    sigma = pyro.sample("sigma", dist.HalfNormal(10.))

    with pyro.plate("data", N):
        pyro.sample("obs", dist.Normal(mu, sigma), obs=y)

if __name__ == "__main__":
    nuts_kernel = NUTS(model)
    mcmc = MCMC(nuts_kernel, num_samples=1000, warmup_steps=1000, num_chains=2)
    mcmc.run()

    idata = az.from_pyro(mcmc)
    print(az.summary(idata))
