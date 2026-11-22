| Method on Scenario B ($t_f=300$ d, $\theta_f=225^\circ$, $1218$ kg, $I_{\text{sp}}=3100$ s) | Conv Rate [95% Wilson CI] | Fuel Mean ± SD (kg) | Median Runtime (s) | Notes |
| --- | :---: | :---: | ---: | --- |
| 180-Node LGL Collocation (smooth init) | 1/1 | 244.080 | 47.75 | reprop=4.11e-04, 3 switches |
| Frozen Phase A2 PoNN (ZERO retuning, 20 seeds) | 0/20 (0.0%) [0.0%, 16.1%] | — | 13.52 | 8-res Chebyshev CE, L=70->80 |
| Indirect Shooting (random 7-D costates, 20 seeds) | 17/20 (85.0%) [64.0%, 94.8%] | 243.826 ± 0.000 | 7.71 | 20-step rho continuation |
| Hybrid: Shooting warm-started from Frozen PoNN | 20/20 (100.0%) [83.9%, 100.0%] | 243.826 ± 0.000 | 4.73 | lam0 extracted from PoNN |
