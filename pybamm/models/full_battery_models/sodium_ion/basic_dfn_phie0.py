#
# Basic Doyle-Fuller-Newman (DFN) Model
#
import pybamm
from .base_sodium_ion_model import BaseModel


class BasicDFN(BaseModel):
    """Doyle-Fuller-Newman (DFN) model of a sodium-ion battery, from
    :footcite:t:`Chayambuka2022`.
   
    Parameters
    ----------
    name : str, optional
        The name of the model.

    """

    def __init__(self, name="Doyle-Fuller-Newman model"):
        super().__init__(name=name)
        pybamm.citations.register("Chayambuka2022")
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
        c_e_n = pybamm.Variable(
            "Negative electrolyte concentration [mol.m-3]",
            domain="negative electrode",
        )
        c_e_s = pybamm.Variable(
            "Separator electrolyte concentration [mol.m-3]",
            domain="separator",
        )
        c_e_p = pybamm.Variable(
            "Positive electrolyte concentration [mol.m-3]",
            domain="positive electrode",
        )
        # Concatenations combine several variables into a single variable, to simplify
        # implementing equations that hold over several domains
        c_e = pybamm.concatenation(c_e_n, c_e_s, c_e_p)

        # Electrolyte potential
        phi_e_n = pybamm.Variable(
            "Negative electrolyte potential [V]",
            domain="negative electrode",
        )
        phi_e_s = pybamm.Variable(
            "Separator electrolyte potential [V]",
            domain="separator",
        )
        phi_e_p = pybamm.Variable(
            "Positive electrolyte potential [V]",
            domain="positive electrode",
        )
        phi_e = pybamm.concatenation(phi_e_n, phi_e_s, phi_e_p)

        # Electrode potential
        phi_s_n = pybamm.Variable(
            "Negative electrode potential [V]", domain="negative electrode"
        )
        phi_s_p = pybamm.Variable(
            "Positive electrode potential [V]",
            domain="positive electrode",
        )
        # Particle concentrations are variables on the particle domain, but also vary in
        # the x-direction (electrode domain) and so must be provided with auxiliary
        # domains
        c_s_n = pybamm.Variable(
            "Negative particle concentration [mol.m-3]",
            domain="negative particle",
            auxiliary_domains={"secondary": "negative electrode"},
        )
        c_s_p = pybamm.Variable(
            "Positive particle concentration [mol.m-3]",
            domain="positive particle",
            auxiliary_domains={"secondary": "positive electrode"},
        )

        # Constant temperature
        T = param.T_init

        ######################
        # Other set-up
        ######################

        # Current density
        i_cell = param.current_density_with_time

        # Porosity
        # Primary broadcasts are used to broadcast scalar quantities across a domain
        # into a vector of the right shape, for multiplying with other vectors
        eps_n = pybamm.PrimaryBroadcast(
            pybamm.Parameter("Negative electrode porosity"), "negative electrode"
        )
        eps_s = pybamm.PrimaryBroadcast(
            pybamm.Parameter("Separator porosity"), "separator"
        )
        eps_p = pybamm.PrimaryBroadcast(
            pybamm.Parameter("Positive electrode porosity"), "positive electrode"
        )
        eps = pybamm.concatenation(eps_n, eps_s, eps_p)

        # Active material volume fraction (eps + eps_s + eps_inactive = 1)
        eps_s_n = pybamm.Parameter("Negative electrode active material volume fraction")
        eps_s_p = pybamm.Parameter("Positive electrode active material volume fraction")

        # transport_efficiency
        tor = pybamm.concatenation(
            eps_n**param.n.b_e, eps_s**param.s.b_e, eps_p**param.p.b_e
        )
        a_n = 3 * param.n.prim.epsilon_s_av / param.n.prim.R_typ
        a_p = 3 * param.p.prim.epsilon_s_av / param.p.prim.R_typ

        # Interfacial reactions
        # Surf takes the surface value of a variable, i.e. its boundary value on the
        # right side. This is also accessible via `boundary_value(x, "right")`, with
        # "left" providing the boundary value of the left side
        c_s_surf_n = pybamm.surf(c_s_n)
        c_n_ravg = pybamm.r_average(c_s_n)
        c_e_n_avg = pybamm.x_average(c_e_n)
        sto_surf_n = c_s_surf_n / param.n.prim.c_max
        j0_n = param.n.prim.j0(c_e_n_avg, c_s_surf_n, c_n_ravg, T)
        eta_n = phi_s_n - phi_e_n - param.n.prim.U(sto_surf_n, T)

        c_rt_n = c_s_surf_n/c_n_ravg
        c_diff_rt_n = (param.n.prim.c_max - c_s_surf_n)/(param.n.prim.c_max - c_n_ravg)
        c_rt_ne = c_e_n/c_e_n_avg

        Feta_RT_n = 0.5 * param.F * eta_n / (param.R * T)
        j_n = j0_n * (
            c_rt_n * pybamm.exp(param.n.prim.ne * Feta_RT_n)
            - c_diff_rt_n * c_rt_ne * pybamm.exp(-param.n.prim.ne * Feta_RT_n)
        )

        c_s_surf_p = pybamm.surf(c_s_p)
        c_p_ravg = pybamm.r_average(c_s_p)
        c_e_p_avg = pybamm.x_average(c_e_p)
        sto_surf_p = c_s_surf_p / param.p.prim.c_max
        j0_p = param.p.prim.j0(c_e_p_avg, c_s_surf_p, c_p_ravg, T)
        eta_p = phi_s_p - phi_e_p - param.p.prim.U(sto_surf_p, T)

        c_rt_p = c_s_surf_p/c_p_ravg
        c_diff_rt_p = (param.p.prim.c_max - c_s_surf_p)/(param.p.prim.c_max - c_p_ravg)
        c_rt_pe = c_e_p/c_e_p_avg
        Feta_RT_p = 0.5 * param.F * eta_p / (param.R * T)
        j_s = pybamm.PrimaryBroadcast(0, "separator")
        j_p = j0_p * (
            c_rt_p * pybamm.exp(param.p.prim.ne * Feta_RT_p)
            - c_diff_rt_p * c_rt_pe * pybamm.exp(-param.p.prim.ne * Feta_RT_p)
        )

        a_j_n = a_n * j_n
        a_j_p = a_p * j_p
        a_j = pybamm.concatenation(a_j_n, j_s, a_j_p)

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
        self.initial_conditions[c_s_n] = param.n.prim.c_init
        self.initial_conditions[c_s_p] = param.p.prim.c_init
        ######################
        # Current in the solid
        ######################
        sigma_eff_n = param.n.sigma(T) * eps_s_n**param.n.b_s
        i_s_n = -sigma_eff_n * pybamm.grad(phi_s_n)
        sigma_eff_p = param.p.sigma(T) * eps_s_p**param.p.b_s
        i_s_p = -sigma_eff_p * pybamm.grad(phi_s_p)
        # The `algebraic` dictionary contains differential equations, with the key being
        # the main scalar variable of interest in the equation
        # multiply by Lx**2 to improve conditioning
        self.algebraic[phi_s_n] = param.L_x**2 * (pybamm.div(i_s_n) + a_j_n)
        self.algebraic[phi_s_p] = param.L_x**2 * (pybamm.div(i_s_p) + a_j_p)
        self.boundary_conditions[phi_s_n] = {
            # "left": (pybamm.Scalar(0), "Dirichlet"),
            "left": (i_cell / pybamm.boundary_value(-sigma_eff_n, "left"), "Neumann"),
            "right": (pybamm.Scalar(0), "Neumann"),
        }
        self.boundary_conditions[phi_s_p] = {
            "left": (pybamm.Scalar(0), "Neumann"),
            "right": (i_cell / pybamm.boundary_value(-sigma_eff_p, "right"), "Neumann"),
        }
        # Initial conditions must also be provided for algebraic equations, as an
        # initial guess for a root-finding algorithm which calculates consistent initial
        # conditions
        # self.initial_conditions[phi_s_n] = pybamm.Scalar(0)
        self.initial_conditions[phi_s_n] = param.n.prim.U_init
        self.initial_conditions[phi_s_p] = param.p.prim.U_init
        

        ######################
        # Current in the electrolyte
        ######################
        # modified chi in sodium_ion_parameters file 
        i_e = (param.kappa_e(c_e, T) * tor) * (
            param.chiRT_over_Fc(c_e, T) * pybamm.grad(c_e) - pybamm.grad(phi_e)
        )
        # multiply by Lx**2 to improve conditioning
        self.algebraic[phi_e] = param.L_x**2 * (pybamm.div(i_e) - a_j)
        self.boundary_conditions[phi_e] = {
            # "left": (pybamm.Scalar(0), "Neumann"),
            "left": (pybamm.Scalar(0), "Dirichlet"),
            "right": (pybamm.Scalar(0), "Neumann"),
        }
        # self.initial_conditions[phi_e] = -param.n.prim.U_init
        self.initial_conditions[phi_e] = 0
    

        ######################
        # Electrolyte concentration
        ######################
        N_e = -tor * param.D_e(c_e, T) * pybamm.grad(c_e)
        self.rhs[c_e] = (1 / eps) * (
            -pybamm.div(N_e) + (1 - param.t_plus(c_e, T)) * a_j / param.F
        )
        self.boundary_conditions[c_e] = {
            "left": (pybamm.Scalar(0), "Neumann"),
            "right": (pybamm.Scalar(0), "Neumann"),
        }
        self.initial_conditions[c_e] = param.c_e_init

        ######################
        # (Some) variables
        ######################
        phi_s_p_L = pybamm.boundary_value(phi_s_p, "right")
        phi_s_n_0 = pybamm.boundary_value(phi_s_n, "left")
        # voltage = phi_s_p_L - param.R_contact*i_cell*param.A_cc
        voltage = phi_s_p_L - phi_s_n_0 - param.R_contact*i_cell*param.A_cc
        
        # The `variables` dictionary contains all variables that might be useful for
        # visualising the solution of the model
        self.variables = {
            "Negative particle surface concentration [mol.m-3]": c_s_surf_n,
            "Electrolyte concentration [mol.m-3]": c_e,
            "Positive particle surface concentration [mol.m-3]": c_s_surf_p,
            "Current [A]": I,
            "Negative electrode potential [V]": phi_s_n,
            "Electrolyte potential [V]": phi_e,
            "Positive electrode potential [V]": phi_s_p,
            "Voltage [V]": voltage,
            "Time [s]": pybamm.t,
        }
        # Events specify points at which a solution should terminate
        self.events += [
            pybamm.Event("Minimum voltage [V]", voltage - param.voltage_low_cut),
            pybamm.Event("Maximum voltage [V]", param.voltage_high_cut - voltage),
            pybamm.Event(
                "Minimum electrolyte concentration in negative electrode",
                pybamm.min(c_e_n) - 0.1,
            ),
            pybamm.Event(
                "Minimum electrolyte concentration in positive electrode",
                pybamm.min(c_e_p) - 0.1,
            ),

        ]
