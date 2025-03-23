## Column Class

The `Column` class represents a single column (or single _step_) in a PSA process. It sets up the system of equations governing each step, including the differential and algebraic states, boundary conditions, and solver configuration.

```python
from swads.column import Column

column = Column(
    num_spatial_points=20,
    num_components=3,
    num_adsorbents=2,
    num_flow_outputs=1,
    step_inputs=[0.4, 0.6, 0.0, 1.0, 0.02],
    step_config=step_config,
    physical_params=physical_params,
    initial_differential_states=initial_diff_states,
    initial_algebraic_states=initial_alg_states
)
```

### Arguments

- **num_spatial_points** (int): Number of spatial discretization points along the column.
- **num_components** (int): Total number of gas components (including any inert).
- **num_adsorbents** (int): Number of adsorbable components (for adsorbed phase states).
- **num_flow_outputs** (int): Factor controlling how many flow outputs can exit the column (normally `1` for a single outlet).
- **step_inputs** (list or None): Feed conditions `[y1, y2, ..., T_in/T_ref, F_in]` if the step has a feed; otherwise empty or `None`.
- **step_config** (dict): Configuration for this PSA step (including `name`, `boundary_conditions`, pressures, times, etc.).
- **physical_params** (dict): Physical and model parameters (e.g., densities, heat capacities, isotherm info).
- **initial_differential_states** (ndarray): Initial values for the ODE (differential) states.
- **initial_algebraic_states** (ndarray): Initial values for the algebraic (constraint) states.

### Methods

#### `integrate(time_array, integrator_type="idas", integrator_opts=None)`

```python
diff_sol, alg_sol = column.integrate(
    time_array,
    integrator_type="idas",
    integrator_opts=None
)
```

Integrates the DAE system over the given (normalized) time array. Typically, the solver expects `time_array[0] == 0.0` and `time_array[-1] == 1.0`.

**Arguments:**

- **time_array** (ndarray): Array of (normalized) time points for integration from 0 to 1.
- **integrator_type** (str): Type of integrator plugin to use (e.g. `"idas"`, `"cvodes"`). Defaults to `"idas"`.
- **integrator_opts** (dict): Additional options for the chosen integrator.

**Returns:**

A tuple `(diff_sol, alg_sol)`, where:

- `diff_sol` is an array (or list of CasADi DM) holding the values of the differential states at each time in `time_array`.
- `alg_sol` is an array (or list of CasADi DM) holding the values of the algebraic states at each time in `time_array`.

## Cycle Class

The `Cycle` class represents an entire PSA cycle composed of multiple sequential steps (each step is solved by creating and running a `Column` model under the specified boundary conditions).

```python
from swads.cycle import Cycle

simulator = Cycle(
    nPDE=20,
    nt=10,
    file_name="output/PSA.hdf5",
    feed_conditions=[0.4, 0.6, 0.0, 1.0, 0.02],
    physical_params=physical_params,
    step_config=step_configs
)
```

### Arguments

- **nPDE** (int): Number of spatial discretization points.
- **nt** (int): Number of time discretization points per step (the solver integrates from t=0 to t=1, subdivided into `nt` intervals).
- **feed_conditions** (list): Composition and feed flow if you have a feed step, e.g. `[y1, y2, ..., T_in/T_ref, F_in]`.
- **physical_params** (dict): Physical parameters and reference conditions for the model.
- **step_config** (list): List of dictionaries, each describing one step in the PSA cycle (name, pressure profile, boundary conditions, etc.).
- **file_name** (str): Path to the file where results and intermediate states will be saved (`"output/PSA.hdf5"` by default).

### Methods

#### `initialize(p=6.0, T=298.15, d_val=0.5)`

```python
simulator.initialize(p=6.0, T=298.15, d_val=0.5)
```

Initializes the simulator with starting conditions for pressure, temperature, and adsorbent loading. If a state file already exists at `file_name`, it will load those states instead.

**Arguments:**

- **p** (float): Initial absolute pressure to assume for the column (must be consistent with reference units in `physical_params`).
- **T** (float): Initial temperature in K.
- **d_val** (float): Initial value for the dimensionless loading state(s) if not loading from file.

#### `run(css_tol=None)`

```python
results = simulator.run(css_tol=1e-4)
```

Runs the PSA cycle repeatedly until convergence to a cyclic steady state (CSS). Each full cycle loops through all configured steps in `step_config`.

**Arguments:**

- **css_tol** (float or None): Tolerance for cyclic steady-state convergence. If `None`, the default class-level tolerance (`1e-8`) is used.

**Returns:**

A dictionary (`result_dict`) containing:

- Per-step data: e.g. `result_dict["Adsorption"]` with `"x_res"`, `"z_res"`, `"prod"`.
- Aggregated product arrays: `"prod_all"`.
- Final product info for each step (e.g. `"PR_Adsorption"`).
