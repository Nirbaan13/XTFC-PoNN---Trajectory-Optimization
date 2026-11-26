| Ablation Group | Variant | Seeds | Conv (<5e-2) | Median Fuel (kg) | Median ODE RMS | Median Reprop | Median Runtime (s) |
| --- | --- | :---: | :---: | ---: | ---: | ---: | ---: |
| A4(a) Formulation | 8-res CE (Refined L=80) | 10 | 10/10 (100%) | 396.52 | 4.80e-03 | 3.63e-02 | 12.40 |
| A4(a) Formulation | 8-res CE (Coarse L=70) | 10 | 0/10 (0%) | 400.70 | 2.06e-03 | 8.64e-02 | 3.07 |
| A4(a) Formulation | 14-res Unreduced (L=50) | 10 | 1/10 (10%) | 965.96 | 2.20e-02 | 1.10e+00 | 30.11 |
| A4(b) Basis | Chebyshev Shifted (CGL) | 10 | 10/10 (100%) | 396.52 | 4.80e-03 | 3.63e-02 | 12.24 |
| A4(b) Basis | Chebyshev Literal T0..T69 (CGL) | 10 | 9/10 (90%) | 399.45 | 4.86e-03 | 4.39e-02 | 10.51 |
| A4(b) Basis | Random tanh ELM (CGL) | 10 | 0/10 (0%) | 437.00 | 1.42e-02 | 1.03e-01 | 41.11 |
| A4(b) Basis | Random tanh ELM (Uniform) | 10 | 0/10 (0%) | 432.79 | 1.41e-02 | 1.02e-01 | 45.82 |
| A4(c) Schedule | n5_to_1e-10 | 5 | 0/5 (0%) | 524.79 | 4.10e-03 | 6.36e-01 | 9.70 |
| A4(c) Schedule | n10_to_1e-10 | 5 | 0/5 (0%) | 392.65 | 4.89e-03 | 8.01e-02 | 7.50 |
| A4(c) Schedule | n20_to_1e-10 | 5 | 5/5 (100%) | 396.52 | 4.80e-03 | 3.63e-02 | 10.14 |
| A4(c) Schedule | n40_to_1e-10 | 5 | 0/5 (0%) | 398.34 | 4.81e-03 | 1.25e-01 | 23.52 |
| A4(c) Schedule | n20_to_1e-3 | 5 | 0/5 (0%) | 398.81 | 4.64e-03 | 7.74e-02 | 25.77 |
| A4(c) Schedule | n20_to_1e-6 | 5 | 0/5 (0%) | 397.21 | 4.79e-03 | 1.06e-01 | 18.27 |
| A4(d) Grid/Opt | uniform_140_lm_svd | 5 | 0/5 (0%) | NaN | 6.71e-06 | 1.49e+00 | 1.51 |
| A4(d) Grid/Opt | cgl_140_lm_svd | 5 | 0/5 (0%) | 400.70 | 2.06e-03 | 8.64e-02 | 3.02 |
| A4(d) Grid/Opt | cgl_140_scipy_lm | 5 | 0/5 (0%) | 400.70 | 2.06e-03 | 8.64e-02 | 20.01 |
| A4(d) Grid/Opt | cgl_140_scipy_trf | 5 | 0/5 (0%) | NaN | 1.65e-02 | 1.83e+00 | 28.61 |
| A4(d) Grid/Opt | uniform_540_lm_svd | 5 | 4/5 (80%) | 396.52 | 4.80e-03 | 3.63e-02 | 11.22 |
