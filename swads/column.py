import numpy as np
from casadi import SX, MX, DM, vertcat, integrator, exp


class Column:
    """
    A model class representing a single column step in a PSA process.
      - Setting up the symbolic states and parameters
      - Checking sizes of initial conditions
      - Building the ODE and ALG equations
      - Integrating (solving) the system over a given normalized time array
    """

    def __init__(
        self,
        num_spatial_points,
        num_components,
        num_adsorbents,
        num_flow_outputs,
        step_inputs,
        step_config,
        physical_params,
        initial_differential_states,
        initial_algebraic_states,
    ):
        """
        Initialize the Column model.

        Args:
            num_spatial_points (int): Number of spatial discretization points.
            num_components (int): Number of gas components.
            num_adsorbents (int): Number of adsorbable components (for Q).
            num_flow_outputs (int): Multiplicative factor for outflow cross-section.
            step_inputs (list or None): Values for feed flow, composition, T_in, etc.
            step_config (dict): Dictionary with step settings, boundary_conditions, etc.
            physical_params (dict): Dictionary with physical properties and constants.
            initial_differential_states (np.ndarray): Initial guess/values for ODE states.
            initial_algebraic_states (np.ndarray): Initial guess/values for ALG states.
        """
        self.step_config = step_config
        self.bc_dict = step_config.get("boundary_conditions", {})
        self.psa_step = step_config["name"]
        self.num_spatial_points = num_spatial_points
        self.num_components = num_components
        self.num_adsorbents = num_adsorbents
        self.num_flow_outputs = num_flow_outputs

        # Store initial guesses for the differential and algebraic states:
        self.initial_differential_states = initial_differential_states
        self.initial_algebraic_states = initial_algebraic_states

        # Physical parameters
        self._initialize_physical_parameters(physical_params)

        # Symbolic variables: states & parameters, set up ODEs and ALGs
        self._initialize_symbolic_variables()

        # Build the step-specific parameter vector
        self.initial_parameters = vertcat(
            self.step_config["L"],
            self.step_config["d_inner"],
            self.step_config["d_outer"],
            self.step_config["p_step"],
            self.step_config["t_step"],
            self.step_config["dpdt"],
        )
        # Combine them in a single symbolic vector
        self.model_parameters = vertcat(
            self.length, self.R_i, self.R_o, self.P_end, self.tf, self.dpdt
        )

        # Boundary conditions: may add self.F_in as algebraic state or model input
        self._apply_bc(step_inputs)

        # Build PDE derivatives needed
        self._update_spatial_derivatives()

        # Build ODE and ALG equations
        self._initialize_dae_equations()

        # If the user provided step_inputs, store them as initial inputs
        if step_inputs is not None and len(step_inputs) > 0:
            self.initial_inputs = vertcat(*step_inputs)
        else:
            self.initial_inputs = []

        # Check dimensions
        if not isinstance(self.states_differential, (SX, MX)):
            self.states_differential = vertcat(*self.states_differential)
        if not isinstance(self.states_algebraic, (SX, MX)):
            self.states_algebraic = vertcat(*self.states_algebraic)
        if not isinstance(self.model_parameters, (SX, MX)):
            self.model_parameters = vertcat(*self.model_parameters)
        if not isinstance(self.model_inputs, (SX, MX)):
            self.model_inputs = vertcat(*self.model_inputs)

        self.num_differential_vars = self.states_differential.size()[0]
        self.num_algebraic_vars = self.states_algebraic.size()[0]

        # Ensure ODE and ALG equations are CasADi expressions
        if isinstance(self.ode_equations, list):
            self.ode_equations = vertcat(*self.ode_equations)
        if isinstance(self.algebraic_equations, list):
            self.algebraic_equations = vertcat(*self.algebraic_equations)

        # Check sizes of initial conditions vs. symbolic states
        def _size_ok(array_or_sym):
            # If CasADi type, get .size()[0], else np.size
            if isinstance(array_or_sym, (SX, MX, DM)):
                return array_or_sym.size()[0]
            return np.size(array_or_sym)

        # ODE states:
        assert (
            _size_ok(self.initial_differential_states) == self.num_differential_vars
        ), "Initial differential states size mismatch."
        # Parameters:
        assert (
            _size_ok(self.initial_parameters) == self.model_parameters.size()[0]
        ), "Number of parameters mismatch."

        # ODE dimension check
        assert (
            self.ode_equations.size()[0] == self.num_differential_vars
        ), "ODE size does not match states_differential size."

        # ALG dimension check
        assert (
            self.algebraic_equations.size()[0] <= self.num_algebraic_vars
        ), "ALG equations size does not match states_algebraic size."

    def _initialize_physical_parameters(self, params):
        """
        Load necessary physical parameters from the config dictionary.
        """
        self.htc_adsorbent_wall = params["htc_adsorbent_wall"]
        self.htc_wall_environment = params["htc_wall_environment"]
        self.density_wall = params["density_wall"]
        self.heat_capacity_solid = params["heat_capacity_solid"]
        self.heat_capacity_wall = params["heat_capacity_wall"]
        self.ambient_temperature = params["ambient_temperature"]
        self.adsorbent_density = params["adsorbent_density"]
        self.bed_void_fraction = params["bed_void_fraction"]
        self.mass_transfer_coefficients = params["mass_transfer_coefficients"]
        self.langmuir_k0_array = params["langmuir_k0_array"]
        self.langmuir_qmax_array = params["langmuir_qmax_array"]
        self.langmuir_exponent_array = params["langmuir_exponent_array"]
        self.adsorption_enthalpies = params["adsorption_enthalpies"]
        self.universal_gas_constant = params["universal_gas_constant"]
        self.reference_temp = params["reference_temperature"]
        self.reference_pressure = params["reference_pressure"]

        c_pi_array = params["component_heat_capacities"]
        Y_guess = params["component_composition_guess"]
        assert len(Y_guess) == len(c_pi_array)
        # A simple weighted average gas heat capacity
        self.c_pg = sum(Y_guess[i] * c_pi_array[i] for i in range(len(Y_guess)))

        self.Cref = (
            self.reference_pressure / self.universal_gas_constant / self.reference_temp
        )
        self.Qref = [params["reference_q"] for _ in range(self.num_adsorbents)]
        self.Dref = params["reference_d"]
        self.Uref = params["reference_u"]

    def _initialize_symbolic_variables(self):
        """
        Create symbolic variables for PDE states, set initial conditions,
        define parameter variables, etc.
        """

        # Symbolic PDE states
        self.Y_species = [
            SX.sym(self.psa_step + f"_Y_{i}", self.num_spatial_points)
            for i in range(self.num_components)
        ]
        Ci_syms = [
            SX.sym(self.psa_step + f"_Ci_{i}", self.num_spatial_points)
            for i in range(self.num_components)
        ]
        self.Ci = [self.Cref * Ci_syms[i] for i in range(self.num_components)]

        Q_syms = [
            SX.sym(self.psa_step + f"_Q_{i}", self.num_spatial_points)
            for i in range(self.num_adsorbents)
        ]
        self.Q = [self.Qref[i] * Q_syms[i] for i in range(self.num_adsorbents)]
        eq_syms = [
            SX.sym(self.psa_step + f"_eq_{i}", self.num_spatial_points)
            for i in range(self.num_adsorbents)
        ]
        self.eq = [self.Qref[i] * eq_syms[i] for i in range(self.num_adsorbents)]

        sum_dQdt_sym = SX.sym(self.psa_step + f"_sum_dQdt", self.num_spatial_points)
        self.sum_dQdt = self.Qref[0] * sum_dQdt_sym  # Just referencing Qref[0]

        D_sym = SX.sym(self.psa_step + f"_D", self.num_spatial_points)
        self.D = self.Dref * D_sym

        T_sym = SX.sym(self.psa_step + f"_T", self.num_spatial_points)
        self.T = self.reference_temp * T_sym
        TW_sym = SX.sym(self.psa_step + f"_TW", self.num_spatial_points)
        self.TW = self.reference_temp * TW_sym

        P_sym = SX.sym(self.psa_step + f"_P", 1)
        self.P = self.reference_pressure * P_sym

        U_sym = SX.sym(self.psa_step + f"_U", self.num_spatial_points)
        self.U = self.Uref * U_sym

        dH_sym = SX.sym(self.psa_step + f"_dH", self.num_spatial_points)
        self.Href = 1.0  # scaling if needed
        self.dH = self.Href * dH_sym

        # The total outlet flow vector (for post-processing).
        self.outlet_flow_vector = SX.sym(
            self.psa_step + f"_F_out", self.num_components + 1
        )

        # Possibly used as an input or an algebraic state, depending on BC:
        self.F_in = SX.sym("F_in", 1)
        self.Y_in = SX.sym("Y_in", self.num_components)
        self.T_in_sym = SX.sym("T_in", 1)
        self.T_in = self.reference_temp * self.T_in_sym

        # For convenience:
        self.C_in = self.P / (self.universal_gas_constant * self.T_in)

        # Make PDE geometry parameters:
        self.length = SX.sym("length", 1)
        self.R_i = SX.sym("R_i", 1)
        self.R_o = SX.sym("R_o", 1)
        self.P_end = SX.sym("P_end", 1)
        self.tf = SX.sym("Tf", 1)
        self.dpdt = SX.sym("dpdt", 1)

        self.a_w = np.pi * (self.R_o * self.R_o - self.R_i * self.R_i)
        self.A = np.pi * self.R_i * self.R_i

        # Build the symbolic state vectors for the DAE class:
        # (1) Differential states (ODE)
        self.states_differential = vertcat(
            *(Ci_syms + Q_syms + [T_sym, TW_sym, P_sym, self.outlet_flow_vector])
        )
        # (2) Algebraic states (will be extended in _apply_bc if needed)
        self.states_algebraic = vertcat(
            *(self.Y_species + eq_syms + [D_sym, sum_dQdt_sym, U_sym, dH_sym])
        )

        # model_inputs starts empty; will be appended in _apply_bc
        self.model_inputs = SX([])

    def _apply_bc(self, step_inputs):
        """
        A single boundary-condition function that reads self.bc_dict
        and sets up the PDE domain for each variable accordingly.
        """
        bc = self.bc_dict
        has_feed = bc.get("has_feed", False)
        flow_in_position = bc.get("flow_in_position", None)
        reverse_flow_sign = bc.get("reverse_flow_sign", False)
        use_zero_at_start = bc.get("use_zero_at_start", False)
        use_zero_at_end = bc.get("use_zero_at_end", False)
        force_zero_velocity_at_start = bc.get("force_zero_velocity_at_start", False)
        has_extra_F_in = bc.get("has_extra_F_in", False)

        # Decide if F_in is an input or an algebraic state:
        if has_feed and has_extra_F_in:
            # Put F_in in the algebraic states
            self.states_algebraic = vertcat(self.states_algebraic, self.F_in)
            # Y_in, T_in_sym remain as model inputs
            self.model_inputs = vertcat(self.Y_in, self.T_in_sym)
        elif has_feed and not has_extra_F_in:
            # F_in, Y_in, T_in_sym as model inputs
            self.model_inputs = vertcat(self.Y_in, self.T_in_sym, self.F_in)
        else:
            # No feed => no F_in or composition as inputs
            self.model_inputs = SX([])

        # The flow velocity from the feed:
        # U_in = ± F_in / (C_in * cross_section * #_of_flows * bed_void_fraction)
        sign_flow = -1.0 if reverse_flow_sign else 1.0
        self.U_in = sign_flow * (
            self.F_in
            / (self.C_in * self.A * self.num_flow_outputs * self.bed_void_fraction)
        )

        # A small helper to replicate first/last node:
        def replicate_first(x):
            return vertcat(x[0], x)

        def replicate_last(x):
            return vertcat(x, x[-1])

        # Build BC-extended arrays for each species
        self.CC_species = []
        for i in range(self.num_components):
            if flow_in_position == "start":
                if has_feed:
                    bc_value = self.C_in * self.Y_in[i]
                    self.CC_species.append(vertcat(bc_value, self.Ci[i]))
                else:
                    self.CC_species.append(replicate_first(self.Ci[i]))
            elif flow_in_position == "end":
                if has_feed:
                    bc_value = self.C_in * self.Y_in[i]
                    self.CC_species.append(vertcat(self.Ci[i], bc_value))
                else:
                    self.CC_species.append(replicate_last(self.Ci[i]))
            else:
                if use_zero_at_end:
                    self.CC_species.append(replicate_last(self.Ci[i]))
                elif use_zero_at_start:
                    self.CC_species.append(replicate_first(self.Ci[i]))
                else:
                    self.CC_species.append(self.Ci[i])

        # Build BC-extended array for T
        if flow_in_position == "start":
            if has_feed:
                self.TT_concat = vertcat(self.T_in, self.T)
            else:
                self.TT_concat = replicate_first(self.T)
        elif flow_in_position == "end":
            if has_feed:
                self.TT_concat = vertcat(self.T, self.T_in)
            else:
                self.TT_concat = replicate_last(self.T)
        else:
            if use_zero_at_end:
                self.TT_concat = replicate_last(self.T)
            elif use_zero_at_start:
                self.TT_concat = replicate_first(self.T)
            else:
                self.TT_concat = self.T

        # Build BC-extended array for U
        if flow_in_position == "start":
            if has_feed:
                self.UU_concat = vertcat(self.U_in, self.U)
            else:
                if use_zero_at_start:
                    self.UU_concat = vertcat(0, self.U)
                else:
                    self.UU_concat = replicate_first(self.U)
        elif flow_in_position == "end":
            if has_feed:
                self.UU_concat = vertcat(self.U, self.U_in)
            else:
                if use_zero_at_end:
                    self.UU_concat = vertcat(self.U, 0)
                else:
                    self.UU_concat = replicate_last(self.U)
        else:
            if use_zero_at_end:
                self.UU_concat = vertcat(self.U, 0)
            elif use_zero_at_start:
                self.UU_concat = vertcat(0, self.U)
            else:
                self.UU_concat = vertcat(self.U, 0)

        # Store a flag for forcing U[0]=0
        self.force_zero_velocity_at_start = force_zero_velocity_at_start

    def _update_spatial_derivatives(self):
        """
        Construct dX/dz for velocity, temperature, species, etc.
        """
        self.dUdz_concat = self._dXdz(self.UU_concat)
        self.dTdz_concat = self._dXdz(self.TT_concat)

        # For species, we do d( U * Ci )/dz
        self.dUCidz_species = []
        for j in range(self.num_components):
            UCj = self.UU_concat * self.CC_species[j]
            self.dUCidz_species.append(self._dXdz(UCj))

    def _initialize_dae_equations(self):
        """
        Build the ODE and ALG equations using standard PDE approach
        with boundary conditions built into self.UU_concat, self.CC_species, etc.
        """
        bc = self.bc_dict
        flow_out_position = bc.get("flow_out_position", None)

        # PDE multipliers
        f_solid = (
            self.adsorbent_density
            * (1 - self.bed_void_fraction)
            / self.bed_void_fraction
        )
        C_local = self.P / (self.universal_gas_constant * self.T)
        C_sum_species = sum(self.Ci)

        # ODE for Q: dQ/dt = k * (eq - Q)
        ODE_Q_list = []
        for j in range(self.num_adsorbents):
            rate_j = self.mass_transfer_coefficients[j] * (self.eq[j] - self.Q[j])
            ODE_Q_list.append(self.tf / self.Qref[j] * rate_j)

        # ODE for Ci
        ODE_Ci_species = []
        for i in range(self.num_components):
            if i < self.num_adsorbents:
                eqn_Ci = (
                    -self.tf / self.Cref * self.dUCidz_species[i]
                    - f_solid * self.Qref[i] * ODE_Q_list[i] / self.Cref
                )
            else:
                eqn_Ci = -self.tf / self.Cref * self.dUCidz_species[i]
            ODE_Ci_species.append(eqn_Ci)

        # ODE for T
        ODE_T_expr = (
            1.0
            / self.reference_temp
            / (self.c_pg * C_local + f_solid * self.heat_capacity_solid)
            * (
                f_solid * self.dH
                - self.tf
                * 2
                * self.htc_adsorbent_wall
                / (self.R_i * self.bed_void_fraction)
                * (self.T - self.TW)
                - self.tf * self.c_pg * self.U * C_local * self.dTdz_concat
            )
        )

        # ODE for TW (wall temperature)
        ODE_TW_expr = (
            self.tf
            / self.reference_temp
            * (2 * np.pi)
            / (self.heat_capacity_wall * self.density_wall * self.a_w)
            * (
                self.htc_adsorbent_wall * self.R_i * (self.T - self.TW)
                - self.htc_wall_environment
                * self.R_o
                * (self.TW - self.ambient_temperature)
            )
        )

        # ODE for P
        ODE_P_expr = (
            self.tf / self.reference_pressure * self.dpdt * (self.P_end - self.P)
        )

        # ODE for the outlet flow vector
        if flow_out_position == "end":
            # outflow at the last PDE cell
            C_end = self.P / (self.universal_gas_constant * self.T[-1])
            flow_out = self.U[-1] * (
                C_end * self.A * self.num_flow_outputs * self.bed_void_fraction
            )
            ODE_F_vec = [
                self.Y_species[i][-1] * flow_out for i in range(self.num_components)
            ]
            # last entry is T_out / T_ref * flow_out
            ODE_F_vec.append(self.T[-1] / self.reference_temp * flow_out)
        elif flow_out_position == "start":
            # outflow at the first PDE cell
            C_start = self.P / (self.universal_gas_constant * self.T[0])
            flow_out = self.U[0] * (
                C_start * self.A * self.num_flow_outputs * self.bed_void_fraction
            )
            ODE_F_vec = [
                self.Y_species[i][0] * flow_out for i in range(self.num_components)
            ]
            ODE_F_vec.append(self.T[0] / self.reference_temp * flow_out)
        else:
            # No outflow => zero
            ODE_F_vec = [0.0] * (self.num_components + 1)

        # Combine ODEs
        self.ode_equations = vertcat(
            *ODE_Ci_species,
            *ODE_Q_list,
            ODE_T_expr,
            ODE_TW_expr,
            ODE_P_expr,
            *ODE_F_vec,
        )

        # ALG equations
        # 1) Y_species = Ci / sum(Ci)
        AE_Y_species = [
            self.Ci[i] / C_sum_species - self.Y_species[i]
            for i in range(self.num_components)
        ]

        # 2) D = exp(D) - (1 - sum(eq_i / qmax_i))
        AE_D_expr = exp(self.D) - (
            1
            - sum(
                self.eq[i] / self.langmuir_qmax_array[i]
                for i in range(self.num_adsorbents)
            )
        )

        # 3) eq_i / qmax_i = P * Y_i * K_i * exp(alpha_i * D)
        AE_q_expr_list = []
        for i in range(self.num_adsorbents):
            K_i = self.langmuir_k0_array[i] * exp(
                self.adsorption_enthalpies[i] / (self.universal_gas_constant * self.T)
            )
            eq_temp = self.eq[i] / self.langmuir_qmax_array[
                i
            ] - self.P * self.Y_species[i] * K_i * exp(
                self.langmuir_exponent_array[i] * self.D
            )
            AE_q_expr_list.append(eq_temp)

        # 4) sum_dQdt = sum(dQ_i/dt)
        AE_sum_dQdt_expr = self.sum_dQdt - sum(
            [self.Qref[i] * ODE_Q_list[i] for i in range(self.num_adsorbents)]
        )

        # 5) momentum continuity: (1/V) dP/dt + d( U*C )/dz + f_solid sum(dQdt) = 0
        dUCdz_expr = self._dXdz(
            self.UU_concat * (self.P / (self.universal_gas_constant * self.TT_concat))
        )
        ODE_P_local = ODE_P_expr * self.reference_pressure
        ODE_T_local = ODE_T_expr * self.reference_temp
        dCdt_expr = 1.0 / (
            self.universal_gas_constant * self.reference_temp
        ) * ODE_P_local - (C_local * ODE_T_local / self.reference_temp)
        AE_U_expr = (
            dCdt_expr / self.Cref
            + self.tf * dUCdz_expr / self.Cref
            + self.T / self.reference_temp * f_solid * self.sum_dQdt / self.Cref
        )

        # 6) AE for dH: dH - sum( H_i * dQ_i/dt ) = 0
        AE_dH_expr = self.dH - sum(
            [
                self.adsorption_enthalpies[i] * self.Qref[i] * ODE_Q_list[i]
                for i in range(self.num_adsorbents)
            ]
        )

        self.algebraic_equations = vertcat(
            *AE_Y_species,
            *AE_q_expr_list,
            AE_D_expr,
            AE_sum_dQdt_expr,
            AE_U_expr,
            AE_dH_expr / self.Href,
        )

        # If the user wants to force zero velocity at the first PDE node:
        if self.force_zero_velocity_at_start:
            self.algebraic_equations = vertcat(self.algebraic_equations, self.U[0])

    def _dXdz(self, X_array):
        """
        Simple finite-difference approximation on an array X_array
        (already includes boundary).
        """
        n_tot = X_array.size()[0]
        dz = self.length / self.num_spatial_points
        d_list = []
        for i in range(n_tot - 1):
            d_list.append((X_array[i + 1] - X_array[i]) / dz)
        return vertcat(*d_list)

    @property
    def prod(self):
        """
        Returns the final 'product' array for the step.
        By convention:
            [y1, y2, ..., y_{ncomp}, (T_out / T_ref), total_flow].
        If no outflow, returns feed as "product" if has_feed, else zeros.
        """
        bc = self.bc_dict
        flow_out_position = bc.get("flow_out_position", None)
        has_feed = bc.get("has_feed", False)

        if flow_out_position in ["start", "end"]:
            # The ODE for outlet_flow_vector was set accordingly
            total_flow = sum(
                [self.outlet_flow_vector[i] for i in range(self.num_components)]
            )
            fractions = [
                self.outlet_flow_vector[i] / total_flow
                for i in range(self.num_components)
            ]
            # last entry in outlet_flow_vector is T_out / T_ref * flow_out
            Tout_ratio = self.outlet_flow_vector[self.num_components] / total_flow
            return [*fractions, Tout_ratio, total_flow]
        else:
            # No outflow => If we do something like "BackFill", produce feed
            if has_feed:
                total_flow = self.F_in
                fractions = [self.Y_in[i] for i in range(self.num_components)]
                return [*fractions, self.T_in / self.reference_temp, total_flow]
            # Otherwise, no feed and no outflow => zero
            return [0.0] * (self.num_components + 2)

    def integrate(self, time_array, integrator_type="idas", integrator_opts=None):
        """
        Integrate (solve) the DAE system over the specified normalized time array
        in one shot, using CasADi's built-in multi-output-time feature.

        Args:
            time_array (np.ndarray): Discretized time, typically from 0 to 1 (normalized).
                Must have time_array[0] == 0.0 and time_array[-1] == 1.0
            integrator_type (str): Which CasADi integrator plugin to use (e.g., 'idas', 'cvodes').
            integrator_opts (dict): Options passed directly to the integrator.

        Returns:
            diff_sol, alg_sol : Each is either
                - a NumPy array of shape (len(time_array), #diff or #alg states), or
                - a list of CasADi DM if your initial states are symbolic.

            The row i corresponds to time_array[i].
        """
        if integrator_opts is None:
            integrator_opts = {}

        # Basic checks
        assert len(time_array) > 0, "time_array cannot be empty."
        assert abs(time_array[0]) < 1e-14, "First time must be 0.0"
        assert abs(time_array[-1] - 1.0) < 1e-14, "Last time must be 1.0 (normalized)."

        # If there's only t=0 in time_array, no actual integration is needed
        if len(time_array) == 1:
            # Return just the initial states
            if isinstance(self.initial_differential_states, (SX, MX, DM)):
                diff_sol = [self.initial_differential_states]
                alg_sol = [self.initial_algebraic_states]
            else:
                diff_sol = np.array([self.initial_differential_states])
                alg_sol = np.array([self.initial_algebraic_states])
            return diff_sol, alg_sol

        # Build the DAE dictionary
        dae_dict = {
            "x": self.states_differential,
            "z": self.states_algebraic,
            "p": vertcat(self.model_inputs, self.model_parameters),
            "ode": self.ode_equations,
            "alg": self.algebraic_equations,
        }

        # We'll call the 5-argument integrator:
        #   integrator("name", solver, dae_dict, t0, tout, opts)
        # where t0 = time_array[0] = 0, and tout = time_array[1..end].
        # That way, CasADi will give us solutions at each entry in tout.
        t0 = time_array[0]  # should be 0
        tout = time_array[1:]  # times after 0

        # Build the integrator:
        F = integrator(
            "integrator_fn",  # name
            integrator_type,  # e.g. "idas"
            dae_dict,
            t0,
            tout,  # a list/array of times
            integrator_opts,
        )

        # Evaluate the integrator once, from 0 up to each time in 'tout'.
        sol = F(
            x0=self.initial_differential_states,
            z0=self.initial_algebraic_states,
            p=vertcat(self.initial_inputs, self.initial_parameters),
        )

        # F returns:
        #  - "xf" : the differential states at each of the times in 'tout'
        #  - "zf" : the algebraic states at each of the times in 'tout'
        #
        # Typically, xf.shape == (#diff_states, len(tout)).
        # We'll convert these to NumPy for convenience (unless your initial states were symbolic).

        xf_all = sol["xf"]
        zf_all = sol["zf"]

        # Decide how to store outputs:
        if isinstance(self.initial_differential_states, (SX, MX, DM)):
            # Symbolic or DM -> store in a list of DM
            diff_sol = []
            alg_sol = []

            # Insert the initial condition as the first entry
            diff_sol.append(self.initial_differential_states)
            alg_sol.append(self.initial_algebraic_states)

            # Then each column of xf_all, zf_all is a time snapshot
            n_times = len(tout)
            for i in range(n_times):
                # xf_all[:, i] is the solution at tout[i]
                diff_sol.append(xf_all[:, i])
                alg_sol.append(zf_all[:, i])

        else:
            # Numeric -> use NumPy arrays
            xf_np = xf_all.full()  # shape (#diff_states, len(tout))
            zf_np = zf_all.full()

            # We'll create arrays sized (len(time_array), # of states)
            diff_sol = np.zeros((len(time_array), self.num_differential_vars))
            alg_sol = np.zeros((len(time_array), self.num_algebraic_vars))

            # The 0th row is the initial condition
            diff_sol[0, :] = self.initial_differential_states
            alg_sol[0, :] = self.initial_algebraic_states

            # Fill subsequent rows from columns of xf_np, zf_np
            for i in range(1, len(time_array)):
                diff_sol[i, :] = xf_np[:, i - 1].squeeze()
                alg_sol[i, :] = zf_np[:, i - 1].squeeze()

        return diff_sol, alg_sol
