"""Recovering latent neural-population dynamics from spike counts with a sequence autoencoder.

Ground truth: a 3-D Lorenz system drives 60 simulated neurons through a log-linear map with Poisson
spiking (a standard LFADS-style benchmark). A bidirectional-GRU encoder maps spike-count sequences to
a 3-D latent trajectory; a linear-exp decoder predicts firing rates; training minimises Poisson NLL.
Baselines: PCA and Gaussian-smoothed PCA. Metrics (held-out trials): R^2 of a linear map from the
inferred latents to the true Lorenz state, and R^2 between predicted and true firing rates.
"""
import json, numpy as np, torch, torch.nn as nn, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.ndimage import gaussian_filter1d
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression

torch.set_num_threads(2)
NN, TB, LAT = 60, 100, 3          # neurons, time bins, latent dim

def lorenz(t, s, sig=10., rho=28., beta=8 / 3):
    return [sig * (s[1] - s[0]), s[0] * (rho - s[2]) - s[1], s[0] * s[1] - beta * s[2]]

def make_data(n_trials=600, seed=0, gain=1.2):
    rng = np.random.default_rng(seed)
    C = rng.standard_normal((3, NN)) * 0.5 * gain; b = np.log(0.1) * np.ones(NN)   # ~0.1 spikes/bin baseline
    Z, R, S = [], [], []
    for _ in range(n_trials):
        s0 = rng.uniform(-10, 10, 3) + np.array([0, 0, 25])
        sol = solve_ivp(lorenz, [0, 12], s0, t_eval=np.linspace(2, 12, TB), rtol=1e-6)   # discard transient
        z = sol.y.T; z = (z - np.array([0, 0, 23.5])) / np.array([8, 9, 8.5]); Z.append(z)
        rate = np.minimum(np.exp(z @ C + b), 2.0); R.append(rate); S.append(rng.poisson(rate))
    return np.array(Z, np.float32), np.array(R, np.float32), np.array(S, np.float32)

class SeqAE(nn.Module):
    def __init__(self, hid=64):
        super().__init__()
        self.enc = nn.GRU(NN, hid, batch_first=True, bidirectional=True)
        self.lat = nn.Linear(2 * hid, LAT)
        self.dec = nn.Sequential(nn.Linear(LAT, 32), nn.Tanh(), nn.Linear(32, NN))
    def forward(self, x):
        h, _ = self.enc(x); z = self.lat(h); return z, self.dec(z)   # z: (B,T,3), log-rate

def r2(y, yhat): return float(1 - ((y - yhat) ** 2).sum() / ((y - y.mean(0)) ** 2).sum())

def latent_r2(Ztr_inf, Ztr_true, Zte_inf, Zte_true):
    reg = LinearRegression().fit(Ztr_inf.reshape(-1, Ztr_inf.shape[-1]), Ztr_true.reshape(-1, 3))
    return r2(Zte_true.reshape(-1, 3), reg.predict(Zte_inf.reshape(-1, Zte_inf.shape[-1])))

if __name__ == "__main__":
    Z, R, S = make_data(); ntr = 450
    Ztr, Zte, Rtr, Rte, Str, Ste = Z[:ntr], Z[ntr:], R[:ntr], R[ntr:], S[:ntr], S[ntr:]
    torch.manual_seed(0); model = SeqAE(); opt = torch.optim.Adam(model.parameters(), 3e-3, weight_decay=1e-4)
    Xtr, Xte = torch.tensor(Str), torch.tensor(Ste)
    for ep in range(120):
        perm = torch.randperm(ntr)
        for i in range(0, ntr, 32):
            x = Xtr[perm[i:i + 32]]; _, logr = model(x)
            loss = nn.functional.poisson_nll_loss(logr, x, log_input=True)
            opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        if ep % 25 == 0: print(ep, loss.item(), flush=True)
    with torch.no_grad():
        ztr, _ = model(Xtr); zte, logr_te = model(Xte)
    res = {"GRU autoencoder": dict(latent_r2=latent_r2(ztr.numpy(), Ztr, zte.numpy(), Zte),
                                  rate_r2=r2(Rte.reshape(-1, NN), np.exp(logr_te.numpy()).reshape(-1, NN)))}
    for name, sm in (("PCA (raw counts)", 0), ("PCA (smoothed, sigma=3 bins)", 3)):
        f = lambda A: gaussian_filter1d(A, sm, axis=1) if sm else A
        pca = PCA(3).fit(f(Str).reshape(-1, NN))
        itr = pca.transform(f(Str).reshape(-1, NN)).reshape(ntr, TB, 3); ite = pca.transform(f(Ste).reshape(-1, NN)).reshape(-1, TB, 3)
        # rate R^2 via ridge from PCA latents to log-free rate (linear decoder)
        rr = LinearRegression().fit(itr.reshape(-1, 3), f(Str).reshape(-1, NN))
        res[name] = dict(latent_r2=latent_r2(itr, Ztr, ite, Zte),
                         rate_r2=r2(Rte.reshape(-1, NN), rr.predict(ite.reshape(-1, 3))))
    print(json.dumps(res, indent=2)); json.dump(res, open("results.json", "w"), indent=2)
    fig = plt.figure(figsize=(8, 3.4))
    reg = LinearRegression().fit(ztr.numpy().reshape(-1, 3), Ztr.reshape(-1, 3)); zh = reg.predict(zte.numpy()[0])
    ax = fig.add_subplot(121, projection="3d"); ax.plot(*Zte[0].T, lw=0.8); ax.set_title("true Lorenz latent", fontsize=8)
    ax = fig.add_subplot(122, projection="3d"); ax.plot(*zh.T, lw=0.8, color="C1"); ax.set_title("GRU-inferred (aligned)", fontsize=8)
    plt.tight_layout(); plt.savefig("latents.png", dpi=150)
