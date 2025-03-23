# Theory and Equations

## Introduction to Pressure Swing Adsorption

Pressure Swing Adsorption (PSA) separates gas mixtures by exploiting the different adsorption affinities of components on a solid adsorbent. The process operates cyclically, with each column undergoing multiple steps (pressurization, adsorption, depressurization, purge, etc.) to achieve separation.

## Mathematical Model

### Governing Equations

In a general PSA model, one typically solves a set of partial differential-algebraic equations (PDAEs) that include:

1. **Mass balance** for each component
2. **Energy balance** (gas + solid/column wall)
3. **Momentum balance** (which may be simplified or lumped)
4. **Adsorption kinetics** (e.g., linear driving force)
5. **Adsorption equilibrium** (e.g., Langmuir isotherm)

In the provided code, **pressure is treated as uniform** in each step (a single ODE state), whereas other variables (concentrations, temperature, velocity, etc.) may be discretized axially.

### Example Mass Balance

A common 1D mass balance for component $i$ in the gas phase can be written as:

$$
\varepsilon \frac{\partial c_i}{\partial t}
- \frac{\partial (\varepsilon \, u \, c_i)}{\partial z}
- (1-\varepsilon)\,\rho_s \,\frac{\partial q_i}{\partial t} \;=\; 0,
$$

where:

- $c_i$ is the gas-phase concentration of component $i$,
- $u$ is the superficial velocity,
- $\varepsilon$ is the bed void fraction,
- $\rho_s$ is the adsorbent density,
- $q_i$ is the adsorbed-phase concentration.

### Energy Balance

A typical energy balance (assuming pseudo-homogeneous or separate gas/solid temperatures) might be:

$$
\rho_{\text{bulk}} C_p \frac{\partial T}{\partial t}
- \rho_{\text{bulk}} C_p \, u \frac{\partial T}{\partial z}
+ \frac{\partial}{\partial z}\bigl(\lambda \frac{\partial T}{\partial z}\bigr)
- (1-\varepsilon) \rho_s \sum_i (-\Delta H_i) \frac{\partial q_i}{\partial t} = 0,
$$

where $\Delta H_i$ is the heat of adsorption.

### Adsorption Kinetics

The linear driving force (LDF) model:

$$
\frac{\partial q_i}{\partial t}
= k_i \bigl( q_i^{*} - q_i \bigr),
$$

with $q_i^{*}$ given by an isotherm (Langmuir, etc.).

### Multicomponent Langmuir Isotherm

$$
q_i^* \;=\;
\frac{q_{m,i}\, b_i \, p_i}{1 + \sum_{j} b_j\, p_j},
$$

where $q_{m,i}$ is the saturation capacity for component $i$, and $b_i$ is the isotherm constant.

## Numerical Solution

- **Method of Lines**: The axial coordinate $z$ is discretized, turning PDEs into a large system of ODEs/DAEs in time.
- **Uniform Pressure Approximation**: In this example code, a single ODE is used to track the overall column pressure. If a fully distributed momentum balance were used, we would have a PDE for $P(z,t)$ or $u(z,t)$.
- **CasADi**: Automatic differentiation and DAE integrators (e.g., IDAS) are used for efficient solution.

## Model Parameters

The model requires:

- **Adsorbent properties**: density, heat capacity, etc.
- **Adsorption isotherms and kinetics**: constants for Langmuir, mass transfer coefficients, heats of adsorption.
- **Bed geometry**: length, radius (inner/outer).
- **Heat transfer**: conduction/dispersion terms, wall heat transfer, environment.

All of these parameters are supplied via `physical_params` in the code.
