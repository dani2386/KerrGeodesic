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
    """
    Save the current matplotlib figure to disk.

    Parameters
    ----------
    filename : str
        Name of the file (relative to OUTPUT_DIR).

    Notes
    -----
    - Uses tight bounding box to avoid extra whitespace.
    - Clears the current figure after saving to prevent overlap.
    """
    plt.savefig(f"{OUTPUT_DIR}{filename}", dpi=DPI, bbox_inches="tight")
    plt.clf()
    print(f"Successfully saved {filename}")


# ============================================================
# PHYSICAL MODEL
# ============================================================

def model(t, params):
    """
    Gravitational-wave inspiral waveform model (leading-order PN approximation).

    Parameters
    ----------
    t : array_like
        Time array.
    params : list or array
        Model parameters:
        - m1, m2 : component masses
        - phi0   : phase offset
        - A0     : amplitude scaling factor
        - tcoal  : coalescence time

    Returns
    -------
    h(t) : array_like
        Predicted gravitational-wave strain.
    """
    m1, m2, phi0, A0, tcoal = params # In natural unites (same as in kerr_solver.py)

    M = m1 + m2
    mu = (m1 * m2) / M
    chirp_mass = mu**(3./5.) * M**(1./5.)

    tau = np.maximum(tcoal - t, 1e-12)

    # Phase (leading PN)
    phi = -2 * (5 * chirp_mass)**(-5./8.) * tau**(5./8.)

    # Amplitude
    A = A0 * chirp_mass**(5./4.) * tau**(-1./4.)

    return A * np.cos(phi + phi0)


# ============================================================
# LIKELIHOOD
# ============================================================

def log_likelihood_chunked(Solver, run_id, depth, params):
    """
    Compute the log-likelihood of the data given model parameters.

    Parameters
    ----------
    Solver : KerrSolver
        Object providing gravitational-wave data chunks.
    run_id : int/str
        Identifier for the simulation/run.
    depth : int
        Controls how much data is retrieved.
    params : array_like
        Model parameters.

    Returns
    -------
    float
        Total log-likelihood across all data chunks.
    """

    total_logL = 0.0

    for t, h in Solver.get_gw(run_id, depth):
        y = h[0]  # take only the first channel
        yerr = np.ones_like(y) * 1e-9  # placeholder errors, adjust if you have real error info

        y_pred = model(t, params)

        # Calculates total log likelihood
        total_logL += np.sum(
            np.log(1 / np.sqrt(2 * np.pi * yerr**2))
            - 0.5 * ((y - y_pred)**2) / (yerr**2)
        )

    return total_logL


# ============================================================
# MCMC
# ============================================================
def propose_gaussian(p, stdevs):
    """
    Draw a proposal from a Gaussian distribution.

    Parameters
    ----------
    p : array_like
        Current parameter vector.
    stdevs : array_like
        Standard deviations for each parameter.

    Returns
    -------
    array_like
        Proposed new parameter vector.
    """
    return np.random.normal(p, stdevs)


def log_posterior_chunked(Solver, run_id, depth, params):
    """
    Compute log-posterior = log-likelihood + log-prior.

    Notes
    -----
    - Uses flat (uniform) priors with simple physical constraints.
    - Returns -inf if parameters are unphysical.
    """
    m1, m2, phi0, A0, tcoal = params
    if m1 < 0 or m2 < 0 or A0 < 0 or tcoal < 0: # Invalidades negative values for positive variables
        return -np.inf

    return log_likelihood_chunked(Solver, run_id, depth, params)



def mcmc_chunked(Solver, run_id, depth, logprob, proposal, proposal_args, initial, nsteps):
    """
    Basic Metropolis-Hastings MCMC sampler (component-wise updates).

    Parameters
    ----------
    Solver, run_id, depth : see above
    logprob : function
        Function computing log-posterior.
    proposal : function
        Proposal distribution function.
    proposal_args : list
        Standard deviations for proposal steps (per parameter).
    initial : list
        Initial parameter values.
    nsteps : int
        Number of MCMC steps.

    Returns
    -------
    chain : ndarray
        MCMC samples (nsteps x ndim).
    acceptance_rate : float
        Fraction of accepted proposals.

    Notes
    -----
    - Updates one parameter at a time (Gibbs-like MH).
    - Uses log-space acceptance test for numerical stability.
    """

    p = np.array(initial, dtype=float)
    ndim = len(p)

    logp = logprob(Solver, run_id, depth, p)

    chain = []
    accepted = 0
    
    # Starts MCMC routine
    for _ in range(nsteps):
        i = np.random.randint(ndim)

        p_new = p.copy() # Saves calculated data
        p_new[i] = np.random.normal(p[i], proposal_args[i])

        logp_new = logprob(Solver, run_id, depth, p_new)

        if logp_new - logp > np.log(np.random.rand()):
            p = p_new
            logp = logp_new
            accepted += 1 # Counts acceptes steps

        chain.append(p.copy())

    return np.array(chain), accepted / nsteps


# ============================================================
# MCMC PLOTTING
# ============================================================
def plot_traces(chain, labels):
    """
    Plot trace (time evolution) of each parameter in the MCMC chain.

    Useful for diagnosing convergence and mixing.
    """
    ndim = len(labels)
    fig, axes = plt.subplots(1, ndim, figsize=(5 * ndim, 4))

    # Handle case ndim = 1
    if ndim == 1:
        axes = [axes]

    for i in range(ndim):
        axes[i].plot(chain[:, i])
        axes[i].set_title(labels[i])

    save_plot("mcmc_traces.png")


def plot_mcmc_fits(Solver, run_id, depth, chain, burn=1000):
    """
    Overlay MCMC-sampled waveforms on top of the data.

    Parameters
    ----------
    burn : int
        Number of initial samples to discard (burn-in).
    """
    fig, ax = plt.subplots()
    ax.grid(True)

    # Store all chunks for overlaying MCMC fits
    t_chunks = []
    y_chunks = []

    # Plot each GW chunk individually
    for t, h in Solver.get_gw(run_id, depth):
        y = h[0]
        ax.plot(t, y, color="blue", linewidth=0.5)
        t_chunks.append(t)
        y_chunks.append(y)

    # Overlay MCMC sampled fits
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
# FUNCTION TO USE IN main.py
# ============================================================
def run_mcm(Solver, run_id, depth):
    """
    Run full MCMC pipeline:
    - sampling
    - diagnostics
    - plotting
    """
    print("Doing MCMC analysis...")

    labels = ["m1", "m2", "phi0", "A0", "tcoal"]

    # Run MCMC
    chain, acc = mcmc_chunked(
        Solver,
        run_id,
        depth,
        log_posterior_chunked,
        propose_gaussian,
        [0.001, 0.1, 0.1, 5e-6, 0.1],
        [0.001, 1., -np.pi/2., 2e-5, 1750.],
        20000
    )

    print("Acceptance rate:", acc)

    # MCMC plots
    plot_traces(chain, labels)
    plot_mcmc_fits(Solver, run_id, depth, chain)
    print("MCMC Done!")