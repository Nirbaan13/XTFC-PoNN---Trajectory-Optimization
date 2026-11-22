| Configuration | Fuel (kg) | Delta vs Shoot (kg) | Switches (days) | Sequence | ODE RMS | DOP853 Reprop | Energy Err | Runtime (s) |
| --- | ---: | ---: | --- | --- | ---: | ---: | ---: | ---: |
| Paper check (Shooting, Ref. [31]) | 396.0600 | — | [47, 66, 148, 292] (Fig. 5) | ON-OFF-ON-OFF-ON | — | — | — | — |
| Paper check (PoNN Refined, Sec. 4.1) | 396.8500 | +0.7900 | [47, 66, 148, 292] (Fig. 5) | ON-OFF-ON-OFF-ON | — | — | — | — |
| Phase A1 Indirect Shooting (this repo) | 396.0598 | +0.0000 | [46.58, 68.02, 142.72, 290.25] | ON-OFF-ON-OFF-ON | 3.73e-11 | 3.46e-04 | 9.90e-05 | 5.99 |
| 180-Node LGL Collocation (smooth init) | 396.1793 | +0.1195 | [46.79, 67.89, 142.64, 290.24] | ON-OFF-ON-OFF-ON | 0.00e+00 | 5.12e-04 | 8.22e-04 | 30.88 |
| PoNN Coarse (L=70, n=140 CGL) | 400.6966 | +4.6368 | [46.30, 66.54, 144.98, 291.29] | ON-OFF-ON-OFF-ON | 2.06e-03 | 8.64e-02 | 1.61e-02 | 3.58 |
| PoNN Refined (L=80, n=540 uniform) | 396.5215 | +0.4617 | [49.83, 68.59, 141.72, 290.55] | ON-OFF-ON-OFF-ON | 4.80e-03 | 3.63e-02 | 8.24e-03 | 13.87 |
| PoNN Refined (L=80, n=540 CGL) | 397.5304 | +1.4705 | [50.54, 68.87, 141.56, 290.13] | ON-OFF-ON-OFF-ON | 4.04e-03 | 3.87e-02 | 7.53e-03 | 11.35 |
| PoNN Refined (L=80, switch bracket) | 400.6341 | +4.5743 | [51.78, 72.85, 144.98, 291.29] | ON-OFF-ON-OFF-ON | 2.76e-03 | 1.07e-01 | 1.92e-02 | 26.52 |
| PoNN Literal (L=70, n=140 uni, T0..T69, U(0,1)) | 419.0212 | +22.9614 | 7 switches | ON-OFF-ON-OFF-ON-OFF-ON-OFF | 7.42e-05 | 1.36e+00 | 7.71e-01 | 18.04 |
