# Three-state GPCR desensitization model

Code for the receptor model in *[manuscript title, authors, journal, year]*.

**Provided for peer review only.** Copyright (c) 2026 the authors. All rights reserved. No license is granted to use, copy, modify or distribute this code until the manuscript is published.

Receptors move between three states: free (`R_free`), agonist-bound and signalling (`R_complex`), and desensitized (`R_des`):

```
R_free + L  <-> R_complex     kon, koff
R_complex    -> R_des          kdes
R_des        -> R_free         kres
```

The agonist concentration `L(t)` is piecewise constant. That makes the system linear within each phase, so every phase is solved exactly with a matrix exponential. No ODE solver is used. The total receptor `R_t` is conserved.

Two alternate models add a cAMP readout (`d[cAMP]/dt = ksyn_cAMP * R_complex - kdeg_cAMP * cAMP`). One of them also adds receptor synthesis and degradation (`ksyn_R`, `kdeg_mem`, `kdeg_des`).

## Files

| File | Contents |
|---|---|
| `model/receptor_model.py` | Rate matrices, analytical simulators (`simulate_three_state`, `simulate_alt`), steady states, and the constant and square-wave stimulation protocols |
| `model/model_parameters.py` | Fitted rate constants: `DEFAULT_PARAMS` (ASMC) and `DEFAULT_PARAMS_HEK` (HEK-293). Units are M, s |
| `example.py` | Constant vs oscillatory priming followed by a step to a high dose, for the base and alternate models |

## Running

Requires Python 3.9 or later.

```bash
pip install -r requirements.txt
python example.py
```

Expected output:

```
Constant     peak #1 = 0.482   peak #2 = 0.476   (fraction of R_t active)
Oscillatory  peak #1 = 0.482   peak #2 = 0.806   (fraction of R_t active)
Alt. model, turnover=False: peak #2 / peak #1 cAMP = 1.573
Alt. model, turnover=True : peak #2 / peak #1 cAMP = 1.439
```

## Usage

```python
import numpy as np
from model.model_parameters import DEFAULT_PARAMS
from model.receptor_model import simulate_three_state, make_ampshift_phases, R_COMPLEX

t = np.linspace(0, 4 * 3600, 10_000)                     # s
phases = make_ampshift_phases(6e-8, 3e-6, 2.5 * 3600, 1800)  # [(t_start_s, L_M), ...]
active = simulate_three_state(DEFAULT_PARAMS, t, phases)[R_COMPLEX]
```
