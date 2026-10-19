#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Three-state GPCR receptor model, solved analytically.

Every model is linear/affine in the state for piecewise-constant agonist, so each
phase is one matrix exponential. No ODE solver anywhere. Equations live in
rate_matrix / alt_rate_matrix.
"""

import numpy as np

# Row indices. R_complex is the signalling-competent state ("active receptor").
R_FREE, R_COMPLEX, R_DES = 0, 1, 2
CAMP = 3          # fourth row of simulate_alt's output


# ── Analytical three-state simulation ─────────────────────────────────────────

def rate_matrix(L, params):
    """3×3 rate matrix for y = [R_free, R_complex, R_des] at fixed ligand L."""
    kon  = params['kon']
    koff = params['koff']
    kdes = params['kdes']
    kres = params['kres']
    return np.array([
        [-kon * L,  koff,        kres ],
        [ kon * L, -koff - kdes, 0.0  ],
        [ 0.0,     kdes,        -kres ],
    ])


def ec_fraction(fraction, params):
    """Concentration occupying `fraction` of receptors at steady state (Hill=1).

    EC50 = (koff + kdes) / kon, so the answer is EC50 · f / (1 − f).
    """
    if fraction == 0.0:
        return 0.0
    ec50 = (params['koff'] + params['kdes']) / params['kon']
    return ec50 * fraction / (1.0 - fraction)


def periodic_steady_state(peak_amp, trough_conc, period_s, params):
    """Periodic steady state at end of off-phase (= start of on-phase).

    Solves (M − I) y* = 0 for the monodromy M, with the last row replaced by
    sum(y*) = R_t since M − I is singular by receptor conservation.
    """
    from scipy.linalg import expm
    A_on  = rate_matrix(peak_amp,    params)
    A_off = rate_matrix(trough_conc, params)
    half  = period_s / 2.0
    M = expm(A_off * half) @ expm(A_on * half)
    K = M - np.eye(3)
    K[-1] = 1.0
    b = np.zeros(3)
    b[-1] = params['R_t']
    try:
        return np.linalg.solve(K, b)
    except np.linalg.LinAlgError:
        return np.full(3, np.nan)


def constant_steady_state(conc, params):
    """Steady state under constant agonist: A @ y = 0, last row = sum(y) = R_t."""
    A = rate_matrix(conc, params)
    M = A.copy()
    M[-1] = 1.0
    b = np.zeros(3)
    b[-1] = params['R_t']
    try:
        return np.linalg.solve(M, b)
    except np.linalg.LinAlgError:
        return np.full(3, np.nan)


def _eigen_decompose(A):
    """Return (V, d, V_inv) such that expm(A·t) = V · diag(exp(d·t)) · V_inv."""
    d, V = np.linalg.eig(A)
    V_inv = np.linalg.inv(V)
    return V, d, V_inv


def _propagate(eigen, y0, ts):
    """Evaluate expm(A·t) @ y0 for each t in ts. Returns shape (len(ts), 3)."""
    V, d, V_inv = eigen
    c      = V_inv @ y0
    scaled = c * np.exp(np.outer(np.asarray(ts), d))
    return np.real(scaled @ V.T)


def simulate_three_state(params, t_eval_s, phases):
    """Analytical three-state simulation for a piecewise-constant agonist.

    `t_eval_s` in seconds; `phases` is [(t_start_s, L), ...]. Initial condition
    [R_t, 0, 0] applied at t_eval_s[0]. Returns (3, len(t_eval_s)).
    """
    t_eval = np.asarray(t_eval_s, dtype=float)
    t0     = t_eval[0]
    t_end  = t_eval[-1] + 1e-9

    phases_sorted = sorted(phases, key=lambda x: x[0])
    phase_t = [p[0] for p in phases_sorted]
    phase_L = [p[1] for p in phases_sorted]

    def _L_at(t):
        idx = 0
        for i, ts in enumerate(phase_t):
            if ts <= t:
                idx = i
        return phase_L[idx]

    # Build segment list from t0 to t_end
    seg_starts = [t0]
    seg_Ls     = [_L_at(t0)]
    for ts, L in phases_sorted:
        if t0 < ts < t_end:
            seg_starts.append(ts)
            seg_Ls.append(L)

    eigens = [_eigen_decompose(rate_matrix(L, params)) for L in seg_Ls]

    y_init = np.array([float(params['R_t']), 0.0, 0.0])
    seg_y  = [y_init]
    for i in range(len(seg_Ls) - 1):
        dt     = seg_starts[i + 1] - seg_starts[i]
        y_next = _propagate(eigens[i], seg_y[i], [dt])[0]
        seg_y.append(y_next)

    # Assign each sample time to its segment
    sample_seg = np.full(len(t_eval), len(seg_Ls) - 1, dtype=int)
    for i in range(len(seg_Ls) - 1):
        mask = (t_eval >= seg_starts[i]) & (t_eval < seg_starts[i + 1])
        sample_seg[mask] = i

    out = np.empty((3, len(t_eval)))
    for i in range(len(seg_Ls)):
        mask = (sample_seg == i)
        if not mask.any():
            continue
        dts          = t_eval[mask] - seg_starts[i]
        y_traj       = _propagate(eigens[i], seg_y[i], dts)   # (n, 3)
        out[:, mask] = y_traj.T

    return out


# ── Analytical alternate-model simulation ─────────────────────────────────────

def alt_rate_matrix(L, params, turnover=False):
    """4×4 matrix for y = [R_free, R_complex, R_des, cAMP] at fixed ligand L.

    cAMP is a passive readout, so the upper-left 3×3 matches rate_matrix.
    turnover=True adds kdeg_mem / kdeg_des; ksyn_R is affine and handled by
    simulate_alt via state augmentation, not here.
    """
    kon, koff, kdes, kres = params['kon'], params['koff'], params['kdes'], params['kres']
    ksyn_cAMP, kdeg_cAMP  = params['ksyn_cAMP'], params['kdeg_cAMP']

    if not turnover:
        return np.array([
            [-kon * L,  koff,        kres,   0.0       ],
            [ kon * L, -koff - kdes, 0.0,    0.0       ],
            [ 0.0,      kdes,       -kres,   0.0       ],
            [ 0.0,      ksyn_cAMP,   0.0,   -kdeg_cAMP ],
        ])

    kdeg_mem, kdeg_des = params['kdeg_mem'], params['kdeg_des']
    return np.array([
        [-kon * L - kdeg_mem,  koff,                    kres,             0.0       ],
        [ kon * L,            -koff - kdes - kdeg_mem,  0.0,              0.0       ],
        [ 0.0,                 kdes,                   -kres - kdeg_des,  0.0       ],
        [ 0.0,                 ksyn_cAMP,               0.0,             -kdeg_cAMP ],
    ])


def simulate_alt(params, t_eval_s, phases, turnover=False):
    """Analytical 4-state simulation, mirroring simulate_three_state.

    Dynamics are dy/dt = A·y + s, with s = [ksyn_R, 0, 0, 0] when turnover.
    The affine term is exact via state augmentation: ỹ = [y; 1] is homogeneous
    under Ã = [[A, s], [0, 0]], so the same propagator applies.
    Returns (4, len(t_eval_s)).
    """
    t_eval = np.asarray(t_eval_s, dtype=float)
    t0     = t_eval[0]
    t_end  = t_eval[-1] + 1e-9

    phases_sorted = sorted(phases, key=lambda x: x[0])
    phase_t = [p[0] for p in phases_sorted]
    phase_L = [p[1] for p in phases_sorted]

    def _L_at(t):
        idx = 0
        for i, ts in enumerate(phase_t):
            if ts <= t:
                idx = i
        return phase_L[idx]

    seg_starts = [t0]
    seg_Ls     = [_L_at(t0)]
    for ts, L in phases_sorted:
        if t0 < ts < t_end:
            seg_starts.append(ts)
            seg_Ls.append(L)

    # Build the (augmented) generator for each segment.
    def _build(L):
        A = alt_rate_matrix(L, params, turnover=turnover)
        if not turnover:
            return A
        s = np.array([params['ksyn_R'], 0.0, 0.0, 0.0])
        A_aug = np.zeros((5, 5))
        A_aug[:4, :4] = A
        A_aug[:4, 4]  = s
        return A_aug

    eigens = [_eigen_decompose(_build(L)) for L in seg_Ls]

    y_init = np.array([float(params['R_t']), 0.0, 0.0, 0.0])
    if turnover:
        y_init = np.append(y_init, 1.0)   # constant augmentation component

    seg_y = [y_init]
    for i in range(len(seg_Ls) - 1):
        dt     = seg_starts[i + 1] - seg_starts[i]
        y_next = _propagate(eigens[i], seg_y[i], [dt])[0]
        seg_y.append(y_next)

    sample_seg = np.full(len(t_eval), len(seg_Ls) - 1, dtype=int)
    for i in range(len(seg_Ls) - 1):
        mask = (t_eval >= seg_starts[i]) & (t_eval < seg_starts[i + 1])
        sample_seg[mask] = i

    n_state = len(y_init)
    out = np.empty((n_state, len(t_eval)))
    for i in range(len(seg_Ls)):
        mask = (sample_seg == i)
        if not mask.any():
            continue
        dts          = t_eval[mask] - seg_starts[i]
        y_traj       = _propagate(eigens[i], seg_y[i], dts)   # (n, n_state)
        out[:, mask] = y_traj.T

    return out[:4]   # drop the augmentation row if present


def make_ampshift_phases(low_conc, high_conc, initial_phase_duration, delay=480):
    """L = 0 until delay, low_conc until initial_phase_duration, then high_conc."""
    return [(0.0, 0.0),
            (float(delay), float(low_conc)),
            (float(initial_phase_duration), float(high_conc))]


def make_square_wave_phases(amplitude, trough, delay, period,
                            switch_time, target_conc):
    """Square wave between amplitude and trough from delay to switch_time, then
    target_conc. Times in seconds."""
    phases = [(0.0, 0.0)]
    t      = float(delay)
    half   = float(period) / 2.0
    sw     = float(switch_time)
    on     = True
    while t < sw - 1e-9:
        phases.append((t, float(amplitude) if on else float(trough)))
        t  += half
        on  = not on
    phases.append((sw, float(target_conc)))
    return phases


def detect_peak_max(signal, time_points, target_time, window_size=10_000):
    """Max of `signal` within ±window_size samples of the time nearest target_time."""
    idx = np.argmin(np.abs(time_points - target_time))
    lo  = max(0, idx - window_size)
    hi  = min(len(signal), idx + window_size)
    return np.max(signal[lo:hi])


def signed_peak(window, before, return_index=False):
    """Extremum of `window`, max or min according to the step direction.

    A down-step gives a trough, where a blind max() would return the tail of the
    pre-step decay. `before` is the last sample before the step; comparing it to
    the window mean settles the direction.
    """
    idx = int(np.argmax(window)) if before < window.mean() else int(np.argmin(window))
    return (float(window[idx]), idx) if return_index else float(window[idx])


def simulate_ampshift_active(params, t_eval_s, prime_conc, target_conc,
                             initial_phase_duration, delay, period=None,
                             active_idx=R_COMPLEX):
    """Active-receptor trace over the amplitude-shift protocol.

    Priming is constant when `period` is None, else a square wave of that period
    against a zero trough — the only difference between the two arms.
    """
    if period is None:
        phases = make_ampshift_phases(prime_conc, target_conc,
                                      initial_phase_duration, delay)
    else:
        phases = make_square_wave_phases(prime_conc, 0.0, delay, period,
                                         initial_phase_duration, target_conc)
    return simulate_three_state(params, t_eval_s, phases)[active_idx]
