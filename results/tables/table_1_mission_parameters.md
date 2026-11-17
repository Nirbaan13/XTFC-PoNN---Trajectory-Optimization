# Table 1: Initial Mission, Spacecraft, Propulsion, Orbital and Numerical Parameters

| Parameter Category | Parameter | Symbol | Assumed Initial Value / Setting | Unit | Application / Justification |
| --- | --- | --- | --- | --- | --- |
| Mission | Departure body | – | Earth | – | Initial planetary body |
| Mission | Arrival body | – | Mars | – | Terminal planetary body |
| Mission | Mission type | – | Low-thrust interplanetary transfer | – | Common mission definition for all algorithms |
| Mission | Transfer type | – | Earth–Mars heliocentric transfer | – | Same for X-TFC, indirect shooting and collocation |
| Mission | Nominal transfer time | tf | 247.6–248.4 | days | Heliocentric transfer time; nominal target ≈248 days |
| Mission | Reference transfer time | tf,ref | 247.9 | days | Representative common value for numerical initialization |
| Mission | Total mission time | ttot | ≈474 | days | Escape (≈226 days) plus heliocentric (≈248 days) |
| Mission | Propulsion mode | – | Solar-electric ion propulsion | – | Consistent with Dawn-type spacecraft configuration |
| Mission | Eclipse modeling | – | Not included | – | Explicit methodological assumption |
| Spacecraft | Mass at start of heliocentric phase | m0 | 1,218 | kg | Dawn-class mass used at start of heliocentric phase |
| Spacecraft | Xenon available for heliocentric | mXe,0 | ≈300 | kg | Must exceed reported consumption of 274.8–276.1 kg |
| Spacecraft | Final spacecraft mass | mf | ≈941.9–943.2 | kg | Initial mass minus xenon consumption |
| Spacecraft | Propellant | – | Xenon | – | Ion propulsion propellant |
| Spacecraft | Propellant consumption | Δmprop | 274.8–276.1 | kg | Heliocentric phase |
| Spacecraft | Spacecraft dry mass | mdry | ≈918 | kg | m0 − mXe,0 = 1,218 − 300 kg |
| Spacecraft | Solar-array configuration | – | Large deployable solar arrays | – | Consistent with Dawn-type configuration |
| Propulsion | Maximum thrust | Tmax | ≈0.50 | N | Uprated Dawn-derived thruster set |
| Propulsion | Minimum thrust | Tmin | 0 | N | Allows coast/low-thrust control |
| Propulsion | Specific impulse | Isp | ≈3,100 | s | Representative ion-propulsion assumption |
| Propulsion | Standard gravity | g0 | 9.80665 | m/s² | Standard gravitational acceleration |
| Propulsion | Maximum mass-flow rate | ṁmax | ≈1.64 × 10⁻⁵ | kg/s | Tmax / (Isp * g0) |
| Earth Departure | Parking-orbit altitude | hp | ≈500 | km | Initial circular parking orbit |
| Earth Departure | Earth equatorial radius | RE | 6,378.137 | km | Earth gravitational model |
| Earth Departure | Earth gravitational parameter | μE | 398,600.4418 | km³/s² | Earth departure dynamics |
| Earth Departure | Earth J2 coefficient | J2 | 1.08262668 × 10⁻³ | – | Used in J2 acceleration model |
| Earth Departure | Departure perturbation | – | Earth J2 | – | Applied during parking-orbit escape |
| Heliocentric Dynamics | Sun gravitational parameter | μS | 1.32712440018 × 10¹¹ | km³/s² | Main heliocentric gravitational parameter |
| Heliocentric Dynamics | Earth heliocentric orbit radius | rE | ≈1.0 | AU | Initial heliocentric condition |
| Heliocentric Dynamics | Mars heliocentric orbit radius | rM | ≈1.524 | AU | Terminal heliocentric condition |
| Heliocentric Dynamics | Astronomical unit | AU | 149,597,870.7 | km | Distance normalization |
| Solar Radiation Pressure | SRP included | – | Yes | – | Applied during heliocentric transfer |
| Solar Radiation Pressure | Solar pressure at 1 AU | P0 | ≈4.56 × 10⁻⁶ | N/m² | Reference solar radiation pressure |
| Solar Radiation Pressure | Reflectivity coefficient | CR | ≈1.2 | – | Spacecraft optical coefficient |
| Solar Radiation Pressure | Area-to-mass ratio | A/m | ≈0.02 | m²/kg | Large-solar-array configuration |
| Solar Radiation Pressure | SRP acceleration at 1 AU | aSRP,1AU | ≈1.09 × 10⁻⁷ | m/s² | P0 * CR * A/m |
| Solar Radiation Pressure | SRP acceleration at Mars | aSRP,Mars | ≈4.7 × 10⁻⁸ | m/s² | Scaled by 1/r² to 1.524 AU |

### Representative Method Performance (Seed 1000)

| Method | Propellant Used (kg) | Flight Time (days) | Final Mass (kg) | Mars Arrival Residual | Energy Error | Runtime (min) |
| --- | --- | --- | --- | --- | --- | --- |
| X-TFC PoNN | 274.8 | 247.6 | 943.2 | 3.2e-07 | 8.0e-07 | 18.6 |
| Indirect Shooting | 276.1 | 248.4 | 941.9 | 6.5e-07 | 1.2e-06 | 24.0 |
| Pseudospectral Collocation | 275.3 | 247.9 | 942.7 | 4.0e-07 | 6.0e-07 | 12.2 |