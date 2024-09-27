#
# Basic Single Particle Model (SPM)
#
import pybamm
from .base_sodium_ion_model import BaseModel


class BasicSPM(BaseModel):
    """Single Particle Model (SPM) model of a sodium-ion battery, from
    :footcite:t:`Garapativk2023`.
    Complete PDE is solved for concnetration of sodium ion in electrode instead of  2-parameter model in the garapti 2023 for fair comparison between models and to make it closer to SPM form Marquis 2019.

    Parameters
    ----------
    name : str, optional
        The name of the model.
    """

    def __init__(self, name="Single Particle Model"):
        super().__init__({}, name)
        pybamm.citations.register("Garapativk2023")
        # `param` is a class containing all the relevant parameters and functions for
        # this model. These are purely symbolic at this stage, and will be set by the
        # `ParameterValues` class when the model is processed.
        param = self.param

        ######################
        # Variables
        ######################
        # Variables that depend on time only are created without a domain
        Q = pybamm.Variable("Discharge capacity [A.h]")
        # Variables that vary spatially are created with a domain
        c_s_n = pybamm.Variable(
            "X-averaged negative particle concentration [mol.m-3]",
            domain="negative particle",
        )
        c_s_p = pybamm.Variable(
            "X-averaged positive particle concentration [mol.m-3]",
            domain="positive particle",
        )

        # Constant temperature
        T = param.T_init

        ######################
        # Other set-up
        ######################

        # Current density
        i_cell = param.current_density_with_time
        a_n = 3 * param.n.prim.epsilon_s_av / param.n.prim.R_typ
        a_p = 3 * param.p.prim.epsilon_s_av / param.p.prim.R_typ
        j_n = i_cell / (param.n.L * a_n)
        j_p = -i_cell / (param.p.L * a_p)

        ######################
        # State of Charge
        ######################
        I = param.current_with_time
        # The `rhs` dictionary contains differential equations, with the key being the
        # variable in the d/dt
        self.rhs[Q] = I / 3600
        # Initial conditions must be provided for the ODEs
        self.initial_conditions[Q] = pybamm.Scalar(0)

        ######################
        # Particles
        ######################

        # The div and grad operators will be converted to the appropriate matrix
        # multiplication at the discretisation stage
        N_s_n = -param.n.prim.D(c_s_n, T) * pybamm.grad(c_s_n)
        N_s_p = -param.p.prim.D(c_s_p, T) * pybamm.grad(c_s_p)
        self.rhs[c_s_n] = -pybamm.div(N_s_n)
        self.rhs[c_s_p] = -pybamm.div(N_s_p)
        # Surf takes the surface value of a variable, i.e. its boundary value on the
        # right side. This is also accessible via `boundary_value(x, "right")`, with
        # "left" providing the boundary value of the left side
        c_s_surf_n = pybamm.surf(c_s_n)
        c_s_surf_p = pybamm.surf(c_s_p)
        # Boundary conditions must be provided for equations with spatial derivatives
        self.boundary_conditions[c_s_n] = {
            "left": (pybamm.Scalar(0), "Neumann"),
            "right": (
                -j_n / (param.F * pybamm.surf(param.n.prim.D(c_s_n, T))),
                "Neumann",
            ),
        }
        self.boundary_conditions[c_s_p] = {
            "left": (pybamm.Scalar(0), "Neumann"),
            "right": (
                -j_p / (param.F * pybamm.surf(param.p.prim.D(c_s_p, T))),
                "Neumann",
            ),
        }
        # c_n_init and c_p_init are functions of r and x, but for the SPM we
        # take the x-averaged value since there is no x-dependence in the particles
        self.initial_conditions[c_s_n] = pybamm.x_average(param.n.prim.c_init)
        self.initial_conditions[c_s_p] = pybamm.x_average(param.p.prim.c_init)
        # Events specify points at which a solution should terminate
        sto_surf_n = c_s_surf_n / param.n.prim.c_max
        sto_surf_p = c_s_surf_p / param.p.prim.c_max
        self.events += [
            pybamm.Event(
                "Minimum negative particle surface stoichiometry",
                pybamm.min(sto_surf_n) - 0.000001,
            ),
            pybamm.Event(
                "Maximum negative particle surface stoichiometry",
                (1 - 0.001) - pybamm.max(sto_surf_n),
            ),
            pybamm.Event(
                "Minimum positive particle surface stoichiometry",
                pybamm.min(sto_surf_p) - 0.001,
            ),
            pybamm.Event(
                "Maximum positive particle surface stoichiometry",
                (1 - 0.001) - pybamm.max(sto_surf_p),
            ),
        ]

        # Note that the SPM does not have any algebraic equations, so the `algebraic`
        # dictionary remains empty

        ######################
        # (Some) variables
        ######################
        # Interfacial reactions
        RT_F = param.R * T / param.F
        
        # As we are solving with full equation and it is single particle avergae is same as the surface value ( as only one particle is used)
        # But going with the same equation to make it identical to SPMe
        c_n_ravg = c_s_surf_n
        c_p_ravg = c_s_surf_p
        # changing from surf to complete concentration 
        j0_n = param.n.prim.j0(param.c_e_init_av, c_s_surf_n, T)
        j0_p = param.p.prim.j0(param.c_e_init_av, c_s_surf_p, T)
       
        j_r_n = j_n/j0_n
        j_r_p = j_p/j0_p
       
        c_rt_n = c_s_surf_n/c_n_ravg
        c_rt_p = c_s_surf_p/c_p_ravg
        c_diff_rt_n = (param.n.prim.c_max - c_s_surf_n)/(param.n.prim.c_max - c_n_ravg)
        c_diff_rt_p = (param.p.prim.c_max - c_s_surf_p)/(param.p.prim.c_max - c_p_ravg)

        eta_n = (2 / param.n.prim.ne) * RT_F * pybamm.log(
            (j_r_n + pybamm.Sqrt(j_r_n**2 + 4*c_rt_n*c_diff_rt_n)) 
            / (2 * c_rt_n ) 
        )
        eta_p = (2 / param.p.prim.ne) * RT_F * pybamm.log(
            (j_r_p + pybamm.Sqrt(j_r_p**2 + 4*c_rt_p*c_diff_rt_p)) 
            / (2 * c_rt_p ) 
        )
        
        eta_n_avg = pybamm.x_average(eta_n)
        eta_p_avg = pybamm.x_average(eta_p)
   
        phi_s_n = 0

        kappa_e = param.kappa_e(param.c_e_init_av, T)
        eps_n = pybamm.Parameter("Negative electrode porosity")
        eps_p = pybamm.Parameter("Positive electrode porosity")
        eps_s = pybamm.Parameter("Separator porosity")
        # phi_e_r = -(i_cell/(3*kappa_e))*(
        #     (param.n.L/eps_n**param.n.b_e) 
        #     + 3*(param.s.L/eps_s**param.s.b_e) 
        #     + (param.p.L/eps_p**param.p.b_e) 
        # )
        phi_e_r = 0
        
        phi_e = -eta_n_avg - param.n.prim.U(sto_surf_n, T) + phi_e_r
        phi_s_p = eta_p_avg + phi_e + param.p.prim.U(sto_surf_p, T)
        
        V = phi_s_p - param.R_contact*i_cell*param.A_cc

        whole_cell = ["negative electrode", "separator", "positive electrode"]
        # The `variables` dictionary contains all variables that might be useful for
        # visualising the solution of the model
        # Primary broadcasts are used to broadcast scalar quantities across a domain
        # into a vector of the right shape, for multiplying with other vectors
        self.variables = {
            "Discharge capacity [A.h]": Q,
            "Negative particle surface "
            "concentration [mol.m-3]": pybamm.PrimaryBroadcast(
                c_s_surf_n, "negative electrode"
            ),
            "Electrolyte concentration [mol.m-3]": pybamm.PrimaryBroadcast(
                param.c_e_init_av, whole_cell
            ),
            "Positive particle surface "
            "concentration [mol.m-3]": pybamm.PrimaryBroadcast(
                c_s_surf_p, "positive electrode"
            ),
            "Current [A]": I,
            "Negative electrode potential [V]": pybamm.PrimaryBroadcast(
                phi_s_n, "negative electrode"
            ),
            "Electrolyte potential [V]": pybamm.PrimaryBroadcast(phi_e, whole_cell),
            "Positive electrode potential [V]": pybamm.PrimaryBroadcast(
                phi_s_p, "positive electrode"
            ),
            "Voltage [V]": V,
            "X-averaged negative electrode reaction "
            "overpotential [V]": eta_n_avg,
            "X-averaged positive electrode reaction "
            "overpotential [V]": eta_p_avg,
        }
        self.events += [
            pybamm.Event("Minimum voltage [V]", V - param.voltage_low_cut),
            pybamm.Event("Maximum voltage [V]", param.voltage_high_cut - V),
        ]
