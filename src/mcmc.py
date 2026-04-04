import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize
import scipy.stats
import corner
import os
from .kerr_solver import KerrSolver

# ============================================================
# CONFIGURATION
# ============================================================
OUTPUT_DIR = "bin/"
DPI = 300


def save_plot(filename):
    plt.savefig(f"{OUTPUT_DIR}{filename}", dpi=DPI, bbox_inches="tight")
    plt.clf()
    print(f"Successfully saved {filename}")


# ============================================================
# PHYSICAL MODEL
# ============================================================

def model(t, params):
    m1, m2, R0, phi0, A0 = params


    M = m1 + m2
    mu = (m1 * m2) / M

    chirp_mass = mu**(3./5.) * M**(1./5.)

    tau0 = (5./64.) * R0**4 / (M**2 * mu)
    tau = np.maximum(tau0 - t, 1e-20)

    phi = (2) * (5*chirp_mass)**(-5./8.) * tau**(5./8.)
    A = A0 * chirp_mass**(5./4.) * (5./tau)**(1./4.)

    return A * np.cos(phi + phi0)


# ============================================================
# LIKELIHOOD
# ============================================================

def log_likelihood_chunked(Solver, run_id, depth, params):
    total_logL = 0.0

    for t, h in Solver.get_gw(run_id, depth):
        y = h[0]  # take only the first channel
        yerr = np.ones_like(y)  # placeholder errors, adjust if you have real error info

        y_pred = model(t, params)

        total_logL += np.sum(
            np.log(1 / np.sqrt(2 * np.pi * yerr**2))
            - 0.5 * ((y - y_pred)**2) / (yerr**2)
        )

    return total_logL


# ============================================================
# MCMC
# ============================================================
def propose_gaussian(p, stdevs):
    """Gaussian proposal distribution."""
    return np.random.normal(p, stdevs)


def log_posterior_chunked(Solver, run_id, depth, params):
    """Posterior = likelihood + prior (flat prior here)."""
    m1, m2, R0, phi0, A0 = params
    if m1 < 0 or m2 < 0 or R0 < 0 or A0 < 0:
        return -np.inf

    return log_likelihood_chunked(Solver, run_id, depth, params)



def mcmc_chunked(Solver, run_id, depth, logprob, proposal, proposal_args, initial, nsteps):
    p = np.array(initial, dtype=float)
    ndim = len(p)

    logp = logprob(Solver, run_id, depth, p)

    chain = []
    accepted = 0

    for _ in range(nsteps):
        i = np.random.randint(ndim)

        p_new = p.copy()
        p_new[i] = np.random.normal(p[i], proposal_args[i])

        logp_new = logprob(Solver, run_id, depth, p_new)

        if logp_new - logp > np.log(np.random.rand()):
            p = p_new
            logp = logp_new
            accepted += 1

        chain.append(p.copy())

    return np.array(chain), accepted / nsteps


# ============================================================
# MCMC PLOTTING
# ============================================================
def plot_traces(chain, labels):
    """Plot parameter traces."""
    ndim = len(labels)
    fig, axes = plt.subplots(1, ndim, figsize=(5 * ndim, 4))

    # Handle case ndim = 1
    if ndim == 1:
        axes = [axes]

    for i in range(ndim):
        axes[i].plot(chain[:, i])
        axes[i].set_title(labels[i])

    save_plot("mcmc_traces.png")


def plot_gw_chunked(Solver, run_id, depth, filename):
    fig, ax = plt.subplots()
    ax.grid(True)

    for t, h in Solver.get_gw(run_id, depth):
        y = h[0]
        ax.plot(t, y, color="blue")  # customize color / style as needed

    os.makedirs(os.path.dirname(filename), exist_ok=True)
    fig.savefig(filename, dpi=300, bbox_inches="tight")
    plt.clf()
    print(f"Saved GW plot to {filename}")


def plot_mcmc_fits(Solver, run_id, depth, chain, burn=1000):
    fig, ax = plt.subplots()
    ax.grid(True)

    # --- Store all chunks for overlaying MCMC fits ---
    t_chunks = []
    y_chunks = []

    # Plot each GW chunk individually
    for t, h in Solver.get_gw(run_id, depth):
        y = h[0]
        ax.plot(t, y, color="blue", linewidth=0.5)
        t_chunks.append(t)
        y_chunks.append(y)

    # --- Overlay MCMC sampled fits ---
    n_samples = 20
    samples = chain[burn:][np.random.randint(0, len(chain)-burn, n_samples)]
    for params in samples:
        # Plot MCMC fit for each chunk separately
        for t_chunk in t_chunks:
            y_fit = model(t_chunk, params)
            ax.plot(t_chunk, y_fit, color="red", alpha=0.2)

    ax.set_xlabel("t")
    ax.set_ylabel("h[0]")

    # Save the figure
    save_plot("mcmc_fits.png")


# ============================================================
# PLACEHOLDER AS MAIN
# ============================================================
def run_mcm(Solver, run_id, depth):
    print("Doing MCMC analysis...")

    labels = ["m1", "m2", "R0", "phi0", "A0"]

    # Run MCMC
    chain, acc = mcmc_chunked(
        Solver,
        run_id,
        depth,
        log_posterior_chunked,
        propose_gaussian,
        [0.001, 0.1, 10., 0.1, 1e-6],
        [0.001, 1., 10., -np.pi/2., 5e-6],
        50000
    )

    print("Acceptance rate:", acc)

    # MCMC plots
    plot_traces(chain, labels)
    plot_mcmc_fits(Solver, run_id, depth, chain)
    print("MCMC Done!")