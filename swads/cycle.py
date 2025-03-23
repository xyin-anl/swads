import os
import numpy as np
import h5py
from casadi import Function, vertcat
from column import Column


def load_state(file_name):
    with h5py.File(file_name, "r") as file_handle:
        state_dict = {
            "x": np.array(file_handle["x"]),
            "z": np.array(file_handle["z"]),
            "prod": np.array(file_handle["prod"]),
            "nt": int(np.array(file_handle["nt"])),
            "nPDE": int(np.array(file_handle["nPDE"])),
            "xf_dims": np.array(file_handle["xf_dims"]),
            "zf_dims": np.array(file_handle["zf_dims"]),
        }
    print("State loaded from file", file_name)
    return state_dict


def save_state(file_name, xf, zf, prod, nt, nPDE, xf_dims, zf_dims, result_dict=None):
    os.makedirs(os.path.dirname(file_name), exist_ok=True)
    with h5py.File(file_name, "w") as file_handle:
        file_handle.create_dataset("x", data=xf)
        file_handle.create_dataset("z", data=zf)
        file_handle.create_dataset("prod", data=prod)
        file_handle.create_dataset("nt", data=nt)
        file_handle.create_dataset("nPDE", data=nPDE)
        file_handle.create_dataset("xf_dims", data=xf_dims)
        file_handle.create_dataset("zf_dims", data=zf_dims)

        if result_dict is not None:
            for name, data in result_dict.items():
                if isinstance(data, dict):
                    grp = file_handle.create_group(name)
                    if "x_res" in data:
                        grp.create_dataset("x_res", data=data["x_res"])
                    if "z_res" in data:
                        grp.create_dataset("z_res", data=data["z_res"])
                    if "prod" in data:
                        grp.create_dataset("prod", data=data["prod"])
                else:
                    file_handle.create_dataset(name, data=data)


def get_end_state(x_res, z_res, nt, nPDE, xf_dims, zf_dims):
    """
    Utility to slice out the final ODE (xf) and ALG (zf) state
    from arrays x_res, z_res at time index nt.
    """
    if isinstance(x_res, np.ndarray):
        # x_res, z_res are 2D arrays: shape (nt+1, total_x_dim)
        x_idx = 0
        xf_list = []
        for dim in xf_dims:
            if dim == nPDE:
                xf_list.append(x_res[nt, x_idx : x_idx + dim])
            else:
                xf_list.append(x_res[nt, x_idx : x_idx + dim])
            x_idx += dim
        xf = np.hstack(xf_list)

        z_idx = 0
        zf_list = []
        for dim in zf_dims:
            zf_list.append(z_res[nt, z_idx : z_idx + dim])
            z_idx += dim
        zf = np.hstack(zf_list)
    else:
        # If x_res, z_res are lists of CasADi DM or so
        xf = x_res[nt]
        zf = z_res[nt]
    return xf, zf


def get_prod(model_instance, x_res, z_res, nt):
    """
    Evaluate the 'prod' property at each time step.
    """
    # Create a CasADi function to evaluate model_instance.prod
    # which is a list or SX expression.
    prod_expr = model_instance.prod
    if not isinstance(prod_expr, (list, tuple)):
        # Possibly an SX expression. Convert to a tuple to vertcat
        prod_expr = (prod_expr,)

    prod_eval_fn = Function(
        "prod_eval_fn",
        [
            model_instance.states_differential,
            model_instance.states_algebraic,
            model_instance.model_parameters,
            model_instance.model_inputs,
        ],
        [vertcat(*prod_expr)],
    )

    production_list = []
    if isinstance(x_res[0], np.ndarray):
        for i in range(1, nt + 1):
            tmp_val = prod_eval_fn(
                x_res[i],
                z_res[i],
                model_instance.initial_parameters,
                model_instance.initial_inputs,
            )
            production_list.append(np.squeeze(tmp_val.full()))
    else:
        # If x_res is a list of DM or similar
        for i in range(1, nt + 1):
            tmp_val = prod_eval_fn(
                x_res[i],
                z_res[i],
                model_instance.initial_parameters,
                model_instance.initial_inputs,
            )
            production_list.append(tmp_val)
    return production_list


class Cycle:
    """
    A high-level PSA cycle orchestrator. It can run a single-bed PSA
    process over multiple steps, track the final states, and repeat
    until some convergence is reached.
    """

    def __init__(
        self,
        nPDE,
        nt,
        feed_conditions,
        physical_params,
        step_config,
        file_name="output/PSA.hdf5",
    ):
        self.num_spatial_points = nPDE
        self.nt = nt
        self.file_name = file_name
        self.feed_conditions = feed_conditions  # typically len = num_components + 2
        self.physical_params = physical_params
        self.step_config = step_config

        self.num_components = len(self.physical_params["component_composition_guess"])
        self.num_adsorbents = len(self.physical_params["mass_transfer_coefficients"])
        self.num_flow = self.num_components + 1  # e.g. ODE flow vector dimension
        self.num_flow_out = 1

        # Reference dimensions for solver
        self.num_flow_offset = 2 + self.num_components + self.num_adsorbents

        # IDAS integrator tolerance
        self.IDAS_abstol = 1e-8

        # CSS Tolerance
        self.CSS_TOL = 1e-8
        self.CSS_TOL_temp = 1e0

        self.initialized = False
        self.xf = None
        self.zf = None
        self.xf_dims = None
        self.zf_dims = None

    def initialize(self, p, T, d_val):
        """
        Build an initial guess for variables if no prior file is found.
        """
        if self.file_name is not None and os.path.isfile(self.file_name):
            saved_state = load_state(self.file_name)
            self.xf = saved_state["x"]
            self.zf = saved_state["z"]
            self.xf_dims = saved_state["xf_dims"]
            self.zf_dims = saved_state["zf_dims"]
        else:
            # Build a brand-new initial guess
            y_init = self.physical_params["component_composition_guess"]
            qref_vals = [
                self.physical_params["reference_q"] for _ in range(self.num_adsorbents)
            ]
            eq_vals = [
                self.physical_params["reference_eq_val"]
                for _ in range(self.num_adsorbents)
            ]
            p_ref = self.physical_params["reference_pressure"]
            T_ref = self.physical_params["reference_temperature"]
            c_ref = p / p_ref * T_ref / T
            d_ref = self.physical_params["reference_d"]
            nPDE = self.num_spatial_points

            # ODE states (xf):
            # [ Ci (nPDE*#comp), Q (nPDE*#ads), T(nPDE), TW(nPDE), P(1), F_out(#comp+1) ]
            xf_list = []
            # Ci
            for i in range(self.num_components):
                xf_list.append(c_ref * y_init[i] * np.ones(nPDE))
            # Q
            for i in range(self.num_adsorbents):
                xf_list.append(eq_vals[i] / qref_vals[i] * np.ones(nPDE))
            # T, TW
            xf_list.append(np.ones(nPDE))  # T ~ 1 => T_ref
            xf_list.append(np.ones(nPDE))  # TW
            # P
            xf_list.append(np.array([p / p_ref]))
            # F_out (#comp+1)
            xf_list.append(np.zeros(self.num_components + 1))
            self.xf = np.hstack(xf_list)

            # zf (algebraic) states:
            # [ Y(nPDE*#comp), eq(nPDE*#ads), D(nPDE), sum_dQdt(nPDE), U(nPDE), dH(nPDE), (+ optional F_in(1) if used) ]
            zf_list = []
            for i in range(self.num_components):
                zf_list.append(y_init[i] * np.ones(nPDE))
            for i in range(self.num_adsorbents):
                zf_list.append(eq_vals[i] / qref_vals[i] * np.ones(nPDE))
            zf_list.append(d_val / d_ref * np.ones(nPDE))  # D
            zf_list.append(np.zeros(nPDE))  # sum_dQdt
            zf_list.append(np.ones(nPDE))  # U
            zf_list.append(np.zeros(nPDE))  # dH
            self.zf = np.hstack(zf_list)

            # Build dimension lists
            self.xf_dims = []
            # Ci
            for i in range(self.num_components):
                self.xf_dims.append(nPDE)
            # Q
            for i in range(self.num_adsorbents):
                self.xf_dims.append(nPDE)
            # T, TW
            self.xf_dims.append(nPDE)
            self.xf_dims.append(nPDE)
            # P
            self.xf_dims.append(1)
            # F_out
            self.xf_dims.append(self.num_components + 1)

            self.zf_dims = []
            # Y
            for i in range(self.num_components):
                self.zf_dims.append(nPDE)
            # eq
            for i in range(self.num_adsorbents):
                self.zf_dims.append(nPDE)
            # D, sum_dQdt, U, dH
            self.zf_dims.append(nPDE)
            self.zf_dims.append(nPDE)
            self.zf_dims.append(nPDE)
            self.zf_dims.append(nPDE)

        self.initialized = True

    def _reset_flow(self, x_vector):
        """
        Zero out the ODE flow variables in x_vector for the next step.
        The last (#comp+1) entries in x_vector are the flow_out vector.
        """
        n_flow = self.num_components + 1
        offset = 0
        # sum of PDE states:
        offset += self.num_components * self.num_spatial_points  # Ci
        offset += self.num_adsorbents * self.num_spatial_points  # Q
        offset += self.num_spatial_points  # T
        offset += self.num_spatial_points  # TW
        offset += 1  # P is a scalar
        # Now offset points to F_out
        x_vector[offset : offset + n_flow] = 0.0
        return x_vector

    def _build_step_feed(self, step_conf, final_step_prods):
        """
        Returns the feed vector for the step, or [] if no feed.

        We look at boundary_conditions => has_feed, feed_stream, has_extra_F_in.
        If has_feed==False => return [].
        Otherwise, if feed_stream=='INPUT' => use self.feed_conditions.
        Else => use final_step_prods[feed_stream].

        If 'has_extra_F_in'==True => we remove the last entry (the flow) from the
        feed vector because F_in is an algebraic state, not an input.
        """
        bc = step_conf.get("boundary_conditions", {})
        has_feed = bc.get("has_feed", False)
        feed_stream = bc.get("feed_stream", None)
        has_extra_F_in = bc.get("has_extra_F_in", False)

        if not has_feed:
            return []

        # If feed_stream is "INPUT", take top-level feed_conditions
        if feed_stream == "INPUT":
            # self.feed_conditions is typically [y1,...,yN, T_in/Tref, F_in].
            # if has_extra_F_in => remove F_in from this input vector.
            if has_extra_F_in:
                return self.feed_conditions[:-1]  # all but the flow
            else:
                return self.feed_conditions

        # Else feed_stream references a previous step
        if feed_stream is not None and feed_stream in final_step_prods:
            # final_step_prods[feed_stream] is typically [y1,...,yN, T_ratio, flow_total]
            feed_vals = final_step_prods[feed_stream]
            if has_extra_F_in:
                # remove the last entry (the flow), keep y's + T_in
                return feed_vals[:-1]
            else:
                return feed_vals
        # If feed_stream is None or unknown, no feed
        return []

    def _execute_single_step(self, step_config, xf, zf, step_feed):
        """
        Build a Column instance, integrate, and return final states + product.
        step_feed is a list/array of float parameters matching model_inputs.
        """
        model_step = Column(
            num_spatial_points=self.num_spatial_points,
            num_components=self.num_components,
            num_adsorbents=self.num_adsorbents,
            num_flow_outputs=self.num_flow_out,
            step_inputs=step_feed,
            step_config=step_config,
            physical_params=self.physical_params,
            initial_differential_states=xf,
            initial_algebraic_states=zf,
        )

        x_res, z_res = model_step.integrate(
            np.linspace(0, 1, self.nt + 1),
            integrator_opts={"abstol": self.IDAS_abstol},
        )
        x_array = np.array(x_res)
        z_array = np.array(z_res)

        xf_new, zf_new = get_end_state(
            x_array,
            z_array,
            self.nt,
            self.num_spatial_points,
            self.xf_dims,
            self.zf_dims,
        )
        prod_list = get_prod(model_step, x_res, z_res, self.nt)
        return x_array, z_array, xf_new, zf_new, prod_list

    def _execute_single_cycle(self, xf, zf):
        """
        Perform one full cycle over all steps in self.step_config.
        Return (CSS_err, xf_end, zf_end, result_dict).
        """
        product_all_steps = []
        result_dict = {}
        x0_local = xf.copy()
        z0_local = zf.copy()
        final_step_prods = {}

        for step_conf in self.step_config:
            # Possibly add an extra algebraic state or so:
            if "extra_z_preprocess" in step_conf:
                z0_local = step_conf["extra_z_preprocess"](z0_local)

            # Build the feed to pass into this step
            step_feed = self._build_step_feed(step_conf, final_step_prods)

            # Possibly reset the flow vector in x
            if not step_conf.get("do_not_reset_flow", False):
                x0_local = self._reset_flow(x0_local)

            # Run the step
            x_res, z_res, x0_local, z0_local, prod_list = self._execute_single_step(
                step_conf, x0_local, z0_local, step_feed
            )

            step_name = step_conf["name"]
            result_dict[step_name] = {
                "x_res": x_res,
                "z_res": z_res,
                "prod": prod_list,
            }
            final_prod = prod_list[-1]
            final_step_prods[step_name] = final_prod
            product_all_steps.extend(prod_list)

        # Evaluate how close we are to cyclic steady-state
        CSS_err = np.linalg.norm(
            self._reset_flow(xf) - self._reset_flow(x0_local), ord=np.inf
        )

        # Keep track of final data:
        result_dict["prod_all"] = product_all_steps
        for key, val in final_step_prods.items():
            result_dict["PR_" + key] = val

        return CSS_err, x0_local, z0_local, result_dict

    def run(self, css_tol=None):
        """
        Main entry point to iterate the PSA cycle until reaching final state.
        """
        assert self.initialized, "Cycle not initialized"
        CSS_TOL_local = self.CSS_TOL if css_tol is None else css_tol

        print("Simulating PSA cycle")
        print("CSS tolerance:", CSS_TOL_local)

        iteration_cycle = 0
        xf_local = self.xf[:]
        zf_local = self.zf[:]
        result_dict = {}

        CSS_err_local = CSS_TOL_local * 100.0
        while CSS_err_local > CSS_TOL_local:
            iteration_cycle += 1
            CSS_err_local, xf_local, zf_local, result_dict = self._execute_single_cycle(
                xf_local, zf_local
            )

            if iteration_cycle % 10 == 0 or CSS_err_local < CSS_TOL_local:
                print(f"Iteration {iteration_cycle}: CSS error = {CSS_err_local}")

            # Save intermediate or final states
            save_state(
                self.file_name,
                xf_local,
                zf_local,
                np.array(result_dict["prod_all"]),
                self.nt,
                self.num_spatial_points,
                self.xf_dims,
                self.zf_dims,
            )

        print("Simulation complete")
        self.xf = xf_local
        self.zf = zf_local

        # Write final results with detail
        save_state(
            self.file_name,
            xf_local,
            zf_local,
            np.array(result_dict["prod_all"]),
            self.nt,
            self.num_spatial_points,
            self.xf_dims,
            self.zf_dims,
            result_dict,
        )

        return result_dict
