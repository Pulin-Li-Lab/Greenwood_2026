import numpy as np

from model.model_parameters import DEFAULT_PARAMS
from model.receptor_model import (simulate_ampshift_active, simulate_alt,
                                  make_square_wave_phases, CAMP)

PRIME  = 6e-8
TARGET = 3e-6
PERIOD = 3600
DELAY  = 1800
SHIFT  = 2.5 * 3600
t = np.linspace(0, 4.5 * 3600, 100_000)

pre  = (t >= DELAY) & (t < SHIFT)
post = t >= SHIFT

for name, period in [("Constant", None), ("Oscillatory", PERIOD)]:
    active = simulate_ampshift_active(DEFAULT_PARAMS, t, PRIME, TARGET, SHIFT, DELAY,
                                      period=period)
    print(f"{name:12s} peak #1 = {active[pre].max():.3f}   "
          f"peak #2 = {active[post].max():.3f}   (fraction of R_t active)")

phases = make_square_wave_phases(PRIME, 0.0, DELAY, PERIOD, SHIFT, TARGET)
for turnover in (False, True):
    camp = simulate_alt(DEFAULT_PARAMS, t, phases, turnover=turnover)[CAMP]
    print(f"Alt. model, turnover={turnover!s:5s}: "
          f"peak #2 / peak #1 cAMP = {camp[post].max() / camp[pre].max():.3f}")
