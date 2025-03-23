## Basic PSA Simulation

Below is an example of running PSA cycle simulation for a **three-component** feed (two adsorbable species plus one inert, which is given a zero fraction if inert is absent). We will demonstrate:

1. Defining physical and step configuration parameters
2. Initializing a `Cycle` simulator
3. Running the simulation to cyclic steady state
4. Analyzing results from the output file.

```python
import numpy as np
from swads.cycle import Cycle

# Physical parameters
my_params = {
    # Physical constants and material properties
    "universal_gas_constant": 8.314,  # J/(mol·K)
    "bed_void_fraction": 0.46,
    "adsorbent_density": 1060.0,  # kg/m³
    "density_wall": 8000.0,  # kg/m³

    # Adsorption parameters
    "langmuir_k0_array": [5.15e-5, 1.31e-4],
    "langmuir_qmax_array": [4.71e-3, 4.03e-3],
    "langmuir_exponent_array": [1.84, 2.77],
    "mass_transfer_coefficients": [1.58e-2, 3.62e-5],
    "adsorption_enthalpies": [-23387.0, -20650.0],  # J/mol

    # Heat transfer properties
    "component_heat_capacities": [37.12, 28.84, 35.64],  # for 3 components
    "htc_adsorbent_wall": 0.0018,
    "htc_wall_environment": 0.001,
    "heat_capacity_solid": 230.0,  # J/(kg·K)
    "heat_capacity_wall": 110.0,   # J/(kg·K)

    # Reference and initial guesses
    "component_composition_guess": [0.4, 0.6, 0.0],
    "ambient_temperature": 298.15,        # K
    "reference_temperature": 298.15,      # K
    "reference_pressure": 1.3e6,          # Pa (about 13 bar)
    "reference_q": 1e-3,
    "reference_d": 1.0,
    "reference_eq_val": 0.001,
    "reference_d_val": 0.5,
    "reference_u": 10.0,   # m/s
}

# Example step configurations
my_configs = [
    {
        "name": "Adsorption",
        "L": 1.5,
        "d_inner": 0.03,
        "d_outer": 0.035,
        "p_step": 6.0e5,  # Pa (≈6 bar)
        "dpdt": 1.0e4,
        "t_step": 60.0,   # s
        "boundary_conditions": {
            "has_feed": True,
            "feed_stream": "INPUT",       # take feed from top-level feed_conditions
            "flow_in_position": "start",
            "flow_out_position": "end",
            "reverse_flow_sign": False,
            "use_zero_at_start": False,
            "use_zero_at_end": False,
            "has_extra_F_in": False,
        },
    },
    {
        "name": "ProvidePurge",
        "L": 1.5,
        "d_inner": 0.03,
        "d_outer": 0.035,
        "p_step": 1.94e5,
        "dpdt": 1.0e3,
        "t_step": 36.0,
        "boundary_conditions": {
            "has_feed": False,
            "feed_stream": None,
            "flow_in_position": "start",
            "flow_out_position": "end",
            "reverse_flow_sign": False,
            "use_zero_at_start": True,
            "use_zero_at_end": False,
            "has_extra_F_in": False,
        },
    },
    {
        "name": "BlowDown",
        "L": 1.5,
        "d_inner": 0.03,
        "d_outer": 0.035,
        "p_step": 2.0e4,
        "dpdt": 1.0e4,
        "t_step": 12.0,
        "boundary_conditions": {
            "has_feed": False,
            "feed_stream": None,
            "flow_in_position": None,
            "flow_out_position": "start",
            "reverse_flow_sign": False,
            "use_zero_at_start": False,
            "use_zero_at_end": True,
            "has_extra_F_in": False,
        },
    },
    {
        "name": "Purge",
        "L": 1.5,
        "d_inner": 0.03,
        "d_outer": 0.035,
        "p_step": 2.0e4,
        "dpdt": 1.0e4,
        "t_step": 36.0,
        "boundary_conditions": {
            "has_feed": True,
            # feed for purge taken from "ProvidePurge" product
            "feed_stream": "ProvidePurge",
            "flow_in_position": "end",
            "flow_out_position": "start",
            "reverse_flow_sign": True,
            "use_zero_at_start": False,
            "use_zero_at_end": False,
            "has_extra_F_in": False,
        },
    },
    {
        "name": "BackFill",
        "L": 1.5,
        "d_inner": 0.03,
        "d_outer": 0.035,
        "p_step": 4.376e5,
        "dpdt": 1.0e4,
        "t_step": 48.0,
        "extra_z_preprocess": lambda z: np.hstack([z, 0.0]),
        "boundary_conditions": {
            "has_feed": True,
            "feed_stream": "Adsorption",
            "flow_in_position": "end",
            "flow_out_position": None,
            "reverse_flow_sign": False,
            "use_zero_at_start": False,
            "use_zero_at_end": False,
            "has_extra_F_in": True,  # F_in is an algebraic state
            "force_zero_velocity_at_start": True,
        },
    },
]

# Example feed: [y1, y2, y3, T_in/Tref, F_in]
# Here we say 0.4, 0.6, 0.0, temperature ratio=1.0, flow=0.02 mol/s
feed_conditions = [0.4, 0.6, 0.0, 1.0, 0.02]

# Initialize the cycle simulator
simulator = Cycle(
    nPDE=20,
    nt=10,
    file_name="output/PSA.hdf5",
    feed_conditions=feed_conditions,
    physical_params=my_params,
    step_config=my_configs,
)

# Initialize with starting conditions (6 bar, 298 K, d_val=0.5)
simulator.initialize(p=6.0, T=298.15, d_val=0.5)

# Run simulation to cyclic steady state
result_dict = simulator.run()

# Inspect final product from each step
for key, val in result_dict.items():
    if key.startswith("PR_"):
        print(f"Product from step {key} = {val}")
```

## Analyzing Results

After the simulation, the results for each step are stored in `"output/PSA.hdf5"`. Each step has a group named after the step (e.g., `"Adsorption"`), containing:

- `x_res`: The ODE (differential) state trajectories over `nt+1` time points.
- `z_res`: The algebraic state trajectories over `nt+1` time points.

You can load them with:

```python
import h5py
import numpy as np
import matplotlib.pyplot as plt

with h5py.File("output/PSA.hdf5", "r") as f:
    # Example: read the "Adsorption" step results
    ads_data = f["Adsorption"]
    x_res = np.array(ads_data["x_res"])
    z_res = np.array(ads_data["z_res"])

# The shape of x_res is (nt+1, total_ODE_dimension).
# For a case with:
#   num_components = 3
#   num_adsorbents = 2
#   num_spatial_points = 20
#
# The ODE states are laid out as:
#   (1) C1(0..19), C2(0..19), C3(0..19)      => total  3*nPDE
#   (2) Q1(0..19), Q2(0..19)                => total  2*nPDE
#   (3) T(0..19)                            => total  1*nPDE
#   (4) TW(0..19)                           => total  1*nPDE
#   (5) P (scalar)                          => total  1
#   (6) F_out (size = #components + 1 = 4)  => total  4
#
# So total_ODE_dimension = 3*20 + 2*20 + 20 + 20 + 1 + 4 = 145.
#
# The final row x_res[nt] is the final state.

nt = x_res.shape[0] - 1
nPDE = 20

# Retrieve the final solution:
x_final = x_res[nt, :]

# Example slices:
C1 = x_final[0:nPDE]              # concentration of component 1
C2 = x_final[nPDE:2*nPDE]         # concentration of component 2
C3 = x_final[2*nPDE:3*nPDE]       # concentration of component 3
Q1 = x_final[3*nPDE:4*nPDE]
Q2 = x_final[4*nPDE:5*nPDE]
T  = x_final[5*nPDE:6*nPDE]
TW = x_final[6*nPDE:7*nPDE]

# Pressure is stored as a single value (uniform assumption):
pressure = x_final[7*nPDE]

# Next, the outlet flow vector (4 entries for 3 components + temperature ratio):
flow_slice = slice(7*nPDE+1, 7*nPDE+5)  # 4 elements
F_out_array = x_final[flow_slice]

# Similarly, for the algebraic states:
z_final = z_res[nt, :]

# In the same example scenario, the algebraic states are laid out as:
#   Y1(0..19), Y2(0..19), Y3(0..19), eq1(0..19), eq2(0..19),
#   D(0..19), sum_dQdt(0..19), U(0..19), dH(0..19)
#   (+ optional F_in(1) if "has_extra_F_in" is used in that step)

# For instance, the velocity profile:
U_profile = z_final[7*nPDE:8*nPDE]

# Plot an example temperature profile
z_axis = np.linspace(0, 1, nPDE)
plt.plot(z_axis, T, label="Gas Temperature")
plt.plot(z_axis, TW, label="Wall Temperature")
plt.xlabel("Normalized bed position")
plt.ylabel("Temperature [K]")
plt.legend()
plt.show()
```
