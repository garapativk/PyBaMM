#
# Basic Single Particle Model with electrolyte for capacitor  (SPMe)
#
import pybamm
from .base_lithium_ion_model import BaseModel


class BasicSPMeCap(BaseModel):
    """Single Particle Model  with electrolyte dynamics(SPMe) model for a lithium-ion capacitor.

    This class differs from the :class:`pybamm.lithium_ion.SPM` model class in that it
    includes intercalation only at anode and capapcitance at cathode end. 

    Parameters
    ----------
    name : str, optional
        The name of the model.
    """

    def __init__(self, name="Single Particle Model"):
        super().__init__({}, name)
        
        # `param` is a class containing all the relevant parameters and functions for
        # this model. These are purely symbolic at this stage, and will be set by the
        # `ParameterValues` class when the model is processed.
        param = self.param

        ######################
        # Variables
        ######################
        # Variables that depend on time only are created without a domain
        Q = pybamm.Variable("Discharge capacity [A.h]")
        delta_phi_p = pybamm.Variable("Positive electrode surface potential difference [V]")
        # Variables that vary spatially are created with a domain
        c_s_n = pybamm.Variable(
            "X-averaged negative particle concentration [mol.m-3]",
            domain="negative particle",
        )
        c_s_p = pybamm.Variable(
            "X-averaged positive particle concentration [mol.m-3]",
            domain="positive particle",
        )

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

        # Porosity
        eps_n = pybamm.Parameter("Negative electrode porosity")
        eps_p = pybamm.Parameter("Positive electrode porosity")
        eps_s = pybamm.Parameter("Separator porosity")

        eps_nb = pybamm.PrimaryBroadcast(
            eps_n, "negative electrode")
        eps_sb = pybamm.PrimaryBroadcast(
            eps_s, "separator")
        eps_pb = pybamm.PrimaryBroadcast(
            eps_p, "positive electrode")
        eps = pybamm.concatenation(eps_nb, eps_sb, eps_pb)

        # transport_efficiency
        tor = pybamm.concatenation(
            eps_nb**param.n.b_e, 
            eps_sb**param.s.b_e, 
            eps_pb**param.p.b_e
        )
      
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
        # N_s_p = -param.p.prim.D(c_s_p, T) * pybamm.grad(c_s_p)
        N_s_p = 0
        self.rhs[c_s_n] = -pybamm.div(N_s_n)
        # self.rhs[c_s_p] = -pybamm.div(N_s_p)
        self.rhs[c_s_p] = N_s_p
        # Surf takes the surface value of a variable, i.e. its boundary value on the
        # right side. This is also accessible via `boundary_value(x, "right")`, with
        # "left" providing the boundary value of the left side
        c_s_surf_n = pybamm.surf(c_s_n)
        # c_s_surf_p = pybamm.surf(c_s_p)
        c_s_surf_p = c_s_p
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
            # "right": (
            #     -j_p / (param.F * pybamm.surf(param.p.prim.D(c_s_p, T))),
            #     "Neumann",
            # ),
            "right": (pybamm.Scalar(0), "Neumann"),
        }
        # c_n_init and c_p_init are functions of r and x, but for the SPM we
        # take the x-averaged value since there is no x-dependence in the particles
        self.initial_conditions[c_s_n] = pybamm.x_average(param.n.prim.c_init)
        self.initial_conditions[c_s_p] = pybamm.x_average(param.p.prim.c_init)
        # Events specify points at which a solution should terminate
        sto_surf_n = c_s_surf_n / param.n.prim.c_max
        sto_surf_p = c_s_surf_p / param.p.prim.c_max
        

        # Note that the SPM does not have any algebraic equations, so the `algebraic`
        # dictionary remains empty

        ######################
        # (Some) variables
        ######################
        # Interfacial reactions
        RT_F = param.R * T / param.F
        c_e_n_avg = pybamm.x_average(c_e_n)
        j0_n = param.n.prim.j0(c_e_n_avg, c_s_surf_n, T)
        # j0_n = param.n.prim.j0(param.c_e_init_av, c_s_surf_n, T)
        # j0_p = param.p.prim.j0(param.c_e_init_av, c_s_surf_p, T)
        eta_n = (2 / param.n.prim.ne) * RT_F * pybamm.arcsinh(j_n / (2 * j0_n))

        # eta_p = (2 / param.p.prim.ne) * RT_F * pybamm.arcsinh(j_p / (2 * j0_p))
        C_dl = pybamm.Parameter("Positive electrode double-layer capacity [F.m-2]")
        # C_dl = param.p.C_dl
        self.rhs[delta_phi_p] = j_p/C_dl

        V_p_init = pybamm.Parameter("Initial voltage of positive electrode [V]")
        self.initial_conditions[delta_phi_p] = V_p_init  #pybamm.Scalar(3.14) or 3.05

        eta_p = delta_phi_p
        phi_s_n = 0
        phi_e = -eta_n - param.n.prim.U(sto_surf_n, T)
        
        # phi_s_p = eta_p + phi_e + param.p.prim.U(sto_surf_p, T)
        phi_s_p = eta_p + phi_e 
        # V = phi_s_p

        
        ######################
        # Electrolyte concentration
        ######################
        a_j_n = pybamm.PrimaryBroadcast(
            a_n * j_n, "negative electrode"
        )
        a_j_p = pybamm.PrimaryBroadcast(
            a_p * j_p, "positive electrode"
        )
        a_j_s = pybamm.PrimaryBroadcast(0, "separator")
        a_j = pybamm.concatenation(a_j_n, a_j_s, a_j_p)

        N_e = -tor * param.D_e(c_e, T) * pybamm.grad(c_e)

        self.rhs[c_e] = (1 / eps) * (
            -pybamm.div(N_e) 
            + (1 - param.t_plus(c_e, T)) * a_j / param.F
        )

        self.boundary_conditions[c_e] = {
            "left": (pybamm.Scalar(0), "Neumann"),
            "right": (pybamm.Scalar(0), "Neumann"),
        }
        self.initial_conditions[c_e] = param.c_e_init
        
        
        c_e_avg = pybamm.x_average(c_e)
        c_e_n_avg = pybamm.x_average(c_e_n)
        c_e_p_avg = pybamm.x_average(c_e_p)
        eta_c = (
            (1 - param.t_plus(c_e_avg, T))
            *2*RT_F/c_e_avg
            *(c_e_p_avg - c_e_n_avg)
        )

        kappa_e = param.kappa_e(param.c_e_init_av, T)
        phi_e_r = -(i_cell/3*kappa_e)*(
            (param.n.L/eps_n**param.n.b_e) 
            + 3*(param.s.L/eps_s**param.s.b_e) 
            + (param.p.L/eps_p**param.p.b_e) 
        )

        # kappa_e_n = pybamm.x_average(param.kappa_e(c_e_n, T))
        # kappa_e_s = pybamm.x_average(param.kappa_e(c_e_s, T))
        # kappa_e_p = pybamm.x_average(param.kappa_e(c_e_p, T))

        
        # phi_e_r = -(i_cell)*(
        #     (param.n.L/(3*kappa_e_n*eps_n**param.n.b_e)) 
        #     + (param.s.L/(kappa_e_s*eps_s**param.s.b_e)) 
        #     + (param.p.L/(3*kappa_e_p*eps_p**param.p.b_e)) 
        # )

        phi_s_r = -(i_cell/3)*(
            param.n.L/param.n.sigma(T) 
            + param.p.L/param.p.sigma(T)
        )

        
        R_contact_n = 2e-4  # ohm*m2
        # for less than 10C
        # R_contact_n = 0*2e-4  # ohm*m2
        
        R_contact_p = 0*1e-4  # ohm*m2
        # R_contact = 1.5e-4  # ohm*m2
        # V_p = eta_p + phi_e_r + phi_s_r   - R_contact_p*i_cell 
        V_p = eta_p + phi_e_r + phi_s_r + eta_c  - R_contact_p*i_cell 
        # V_p = eta_p + phi_e_r + phi   _s_r + eta_c  
        # V_p = eta_p 
        # V_n = eta_n + param.n.prim.U(sto_surf_n, T) 
        V_n = eta_n + param.n.prim.U(sto_surf_n, T) +  R_contact_n *  i_cell 
        # V_n = eta_n + param.n.prim.U(sto_surf_n, T) +  R_contact_n *  i_cell - eta_c 
        V = V_p - V_n 

        # V_p = eta_p 
        # V_n = param.n.prim.U(sto_surf_n, T) 
        # V = eta_p - eta_n - param.n.prim.U(sto_surf_n, T) + phi_e_r

        soc_cutoff = pybamm.Parameter("SoC cutoff")
        soc_mincutoff = pybamm.Parameter("SoC min cutoff")
        dV_window = pybamm.Parameter("Permissible change in potential of positive electrode [V]")
        
        dV_pos = V_p - V_p_init 

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
            "Electrolyte concentration [mol.m-3]": c_e,
            # "Positive particle surface "
            # "concentration [mol.m-3]": pybamm.PrimaryBroadcast(
            #     c_s_surf_p, "positive electrode"
            # ),
            # "Positive particle surface "
            # "concentration [mol.m-3]":  c_s_surf_p ,
            "Current [A]": I,
            "Negative electrode potential [V]": pybamm.PrimaryBroadcast(
                phi_s_n, "negative electrode"
            ),
            "Electrolyte potential [V]": pybamm.PrimaryBroadcast(phi_e, whole_cell),
            "Positive electrode potential [V]": pybamm.PrimaryBroadcast(
                phi_s_p, "positive electrode"
            ),
            "Voltage [V]": V,
            "Negative electrode surface potential difference [V]": V_n,
            "Positive electrode surface potential difference [V]": V_p,
            "SoC of negative electrode": sto_surf_n,

        }
      
        self.events += [
            pybamm.Event("Minimum voltage [V]", V - param.voltage_low_cut),
            pybamm.Event("Maximum voltage [V]", param.voltage_high_cut - V),
            pybamm.Event("Maximum permissible window of positive electrode", dV_window - dV_pos),
            pybamm.Event(
                "Minimum electrolyte concentration",
                pybamm.min(c_e) - 1e-6,
            ),
            pybamm.Event(
                "Minimum negative particle surface stoichiometry",
                pybamm.min(sto_surf_n) - soc_mincutoff,
                # pybamm.min(sto_surf_n) - 1e-6,
            ),
            pybamm.Event(
                "Maximum negative particle surface stoichiometry",
                soc_cutoff - pybamm.max(sto_surf_n),
                # (1 - 0.01) - pybamm.max(sto_surf_n),
            ),
            # pybamm.Event(
            #     "Minimum positive particle surface stoichiometry",
            #     pybamm.min(sto_surf_p) - 0.01,
            # ),
            pybamm.Event(
                "Maximum positive particle surface stoichiometry",
                (1 - 0.01) - pybamm.max(sto_surf_p),
            ),
        ]
