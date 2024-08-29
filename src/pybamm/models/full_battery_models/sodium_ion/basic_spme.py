#
# Basic Single Particle Model with Electrolyte (SPMe)
#
import pybamm
from .base_sodium_ion_model import BaseModel


class BasicSPMe(BaseModel):
    """SPMe (SPMe) model of a sodium-ion battery, from
    Brosa Planella 2021, Marquius 2019. Adapted for sodium-ion by Garapati.
    Li-ion model Equations from the 
    https://docs.pybamm.org/en/stable/source/examples/notebooks/models/SPMe.html
    :footcite:t:`Garapati2024`.
   
    Parameters
    ----------
    name : str, optional
        The name of the model.

    """
    def __init__(self, name="SPMe model"):
        super().__init__(name=name)
        pybamm.citations.register("Garapati2024")
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

        # Particle concentrations are variables on the particle domain, but also vary in
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

        # Porosity
        # Primary broadcasts are used to broadcast scalar quantities across a domain
        # into a vector of the right shape, for multiplying with other vectors
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
        # transport efficiency
        # tor_n = eps_n**param.n.b_e
        # tor_s = eps_s**param.s.b_e
        # tor_p = eps_p**param.p.b_e
        
        tor_n = eps_nb**param.n.b_e
        tor_s = eps_sb**param.s.b_e
        tor_p = eps_pb**param.p.b_e
        
        tor = pybamm.concatenation(
            tor_n, tor_s, tor_p
        )

        a_n = 3 * param.n.prim.epsilon_s_av / param.n.prim.R_typ
        a_p = 3 * param.p.prim.epsilon_s_av / param.p.prim.R_typ
        j_n = i_cell / (param.n.L * a_n)
        j_p = -i_cell / (param.p.L * a_p)

        j_nb = pybamm.PrimaryBroadcast(j_n, "negative electrode")
        j_pb = pybamm.PrimaryBroadcast(j_p, "positive electrode")
        j_sb = pybamm.PrimaryBroadcast(0, "separator")

        a_j_n = a_n * j_nb
        a_j_p = a_p * j_pb
        a_j = pybamm.concatenation(a_j_n, j_sb, a_j_p)

        # Interfacial reactions
        # Surf takes the surface value of a variable, i.e. its boundary value on the
        # right side. This is also accessible via `boundary_value(x, "right")`, with
        # "left" providing the boundary value of the left side
        c_s_surf_n = pybamm.surf(c_s_n)
        # c_n_ravg = pybamm.r_average(c_s_n)
        c_n_ravg = c_s_surf_n
        
        # c_e_n_avg = pybamm.x_average(c_e_n)
        sto_surf_n = c_s_surf_n / param.n.prim.c_max
        # j0_n = param.n.prim.j0(c_e_n_avg, c_s_surf_n, T)
        j0_n = param.n.prim.j0(c_e_n, c_s_surf_n, T)
        
        c_s_surf_p = pybamm.surf(c_s_p)
        # c_p_ravg = pybamm.r_average(c_s_p)
        c_p_ravg = c_s_surf_p
        
        # c_e_p_avg = pybamm.x_average(c_e_p)
        sto_surf_p = c_s_surf_p / param.p.prim.c_max
        # j0_p = param.p.prim.j0(c_e_p_avg, c_s_surf_p,  T)
        j0_p = param.p.prim.j0(c_e_p, c_s_surf_p,  T)
       
        c_rt_n = c_s_surf_n/c_n_ravg
        c_diff_rt_n = (param.n.prim.c_max - c_s_surf_n)/(param.n.prim.c_max - c_n_ravg)
       

        c_rt_p = c_s_surf_p/c_p_ravg
        c_diff_rt_p = (param.p.prim.c_max - c_s_surf_p)/(param.p.prim.c_max - c_p_ravg)
        
        c_e_n_avg = pybamm.x_average(c_e_n)
        c_e_p_avg = pybamm.x_average(c_e_p)
        c_e_s_avg = pybamm.x_average(c_e_s)
        
        ce_rt_n = c_e_n/c_e_n_avg
        ce_rt_p = c_e_p/c_e_p_avg

        j_r_n = j_n/j0_n
        j_r_p = j_p/j0_p

        RT_F = param.R * T / param.F
       
        eta_n = (2 / param.n.prim.ne) * RT_F * pybamm.log(
            (j_r_n + pybamm.Sqrt(j_r_n**2 + 4*c_rt_n*ce_rt_n*c_diff_rt_n)) 
            / (2 * c_rt_n ) 
        )
        eta_p = (2 / param.p.prim.ne) * RT_F * pybamm.log(
            (j_r_p + pybamm.Sqrt(j_r_p**2 + 4*c_rt_p*ce_rt_p*c_diff_rt_p)) 
            / (2 *  c_rt_p  )
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
        self.initial_conditions[c_s_n] = pybamm.x_average(param.n.prim.c_init)
        self.initial_conditions[c_s_p] = pybamm.x_average(param.p.prim.c_init)

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
        phi_s_n = 0
          
        delta_phi_s = -(i_cell/3)*(
            param.n.L/param.n.sigma(T) 
            + param.p.L/param.p.sigma(T)
        )
        
          
        # # c_e_avg = pybamm.x_average(c_e)
        
        # # c_e_n_ln_avg = pybamm.x_average(pybamm.log(c_e_n))
        # # c_e_p_ln_avg = pybamm.x_average(pybamm.log(c_e_p))
        # # eta_e = (1 - 2*param.t_plus(c_e_avg, T))*RT_F*(c_e_p_ln_avg - c_e_n_ln_avg)
        
        # # log_c_e_n = pybamm.log(c_e_n)
        # # log_c_e_p = pybamm.log(c_e_p)
        # # delta_log_c_e = pybamm.x_average(log_c_e_p) - pybamm.x_average(log_c_e_n)
        
        # # eta_e = (1 - 2*param.t_plus(c_e_avg, T)) * RT_F * delta_log_c_e
        
        # x_n = pybamm.standard_spatial_vars.x_n
        # x_s = pybamm.standard_spatial_vars.x_s
        # x_p = pybamm.standard_spatial_vars.x_p
        
        
        # def _derivative_macinnes_function(x):
        #     "Compute the derivative of the MacInnes function."
        #     tol = pybamm.settings.tolerances["macinnes__c_e"]
        #     x = pybamm.maximum(x, tol)
        #     return 1 / x
        
        # eta_c_p = -RT_F * pybamm.IndefiniteIntegral(
        #     param.chi(c_e_p, T)
        #     * _derivative_macinnes_function(c_e_p)
        #     * pybamm.grad(c_e_p),
        #     x_p,
        # )
        
        # eta_c_s = -RT_F * pybamm.IndefiniteIntegral(
        #     param.chi(c_e_s, T)
        #     * _derivative_macinnes_function(c_e_s)
        #     * pybamm.grad(c_e_s),
        #     x_s,
        # )
        
        # eta_c_n = -RT_F * pybamm.IndefiniteIntegral(
        #         param.chi(c_e_n, T)
        #         * _derivative_macinnes_function(c_e_n)
        #         * pybamm.grad(c_e_n),
        #         x_n,
        # )
  
        # eta_c_av = (
        #     -pybamm.x_average(eta_c_p)
        #     + pybamm.x_average(eta_c_n)
        #     - pybamm.boundary_value(eta_c_s, "right")
        #     - pybamm.boundary_value(eta_c_n, "right")
        # )
        c_e_avg = pybamm.x_average(c_e)
        chi_av = param.chi(c_e_avg, T)
        
        def _higher_order_macinnes_function(x):
            "Function to differentiate between composite and first-order models"
            tol = pybamm.settings.tolerances["macinnes__c_e"]
            x = pybamm.maximum(x, tol)
            return pybamm.log(x)
        
        macinnes_c_e_p = pybamm.x_average(
            _higher_order_macinnes_function(c_e_p / c_e_n_avg)
        )
        
        macinnes_c_e_n = pybamm.x_average(
            _higher_order_macinnes_function(c_e_n / c_e_n_avg)
        )
         
        eta_e = chi_av * RT_F * (macinnes_c_e_p - macinnes_c_e_n)
  
      
                
        # i_boundary_cc = pybamm.PrimaryBroadcast(
        #     i_cell, "current collector"
        # )
        # L_x = param.L_x
        L_n = param.n.L
        L_p = param.p.L
        L_s = param.s.L
          
        # x_p_edge = pybamm.standard_spatial_vars.x_p_edge
        # x_n_edge = pybamm.standard_spatial_vars.x_n_edge
                
        # i_e_p_edge = i_boundary_cc * (L_x - x_p_edge) / L_p
        # i_e_s_edge = pybamm.PrimaryBroadcastToEdges(i_boundary_cc, "separator")
        # i_e_n_edge = i_boundary_cc * x_n_edge / L_n
        
        # delta_phi_e_p = pybamm.IndefiniteIntegral(
        #     i_e_p_edge / (param.kappa_e(c_e_p, T) * tor_p), x_p
        # )
        
        # delta_phi_e_n = pybamm.IndefiniteIntegral(
        #         i_e_n_edge / (param.kappa_e(c_e_n, T) * tor_n), x_n
        # )
        
        # delta_phi_e_s = pybamm.IndefiniteIntegral(
        #     i_e_s_edge / (param.kappa_e(c_e_s, T) * tor_s), x_s
        # )     
        
                    
        # delta_phi_e_av = (
        #     - pybamm.Integral(delta_phi_e_p, x_p)
        #     + pybamm.Integral(delta_phi_e_n, x_n)
        #     - pybamm.boundary_value(delta_phi_e_s, "right")
        #     - pybamm.boundary_value(delta_phi_e_n, "right")
        # )
        
        # bulk conductivities
        tor_n_av = pybamm.x_average(tor_n)
        tor_s_av = pybamm.x_average(tor_s)
        tor_p_av = pybamm.x_average(tor_p)
                
        kappa_n_av = param.kappa_e(c_e_n_avg, T) * tor_n_av
        kappa_s_av = param.kappa_e(c_e_s_avg, T) * tor_s_av
        kappa_p_av = param.kappa_e(c_e_p_avg, T) * tor_p_av
        
        delta_phi_e =  -(i_cell) * (
             L_n / (3 * kappa_n_av) + L_s / kappa_s_av + L_p / (3 * kappa_p_av)
        )
             
        eta_p_avg = pybamm.x_average(eta_p)
        eta_n_avg = pybamm.x_average(eta_n)
        
        eta_rxn = eta_p_avg - eta_n_avg
        
        phi_e = eta_e + delta_phi_e  - param.n.prim.U(sto_surf_n, T)
    
        phi_s_p = (param.p.prim.U(sto_surf_p, T)  
                   + eta_rxn
                   + phi_e    
                   +  delta_phi_s)        
        
        V = (param.p.prim.U(sto_surf_p, T)  - param.n.prim.U(sto_surf_n, T)  
                   + eta_rxn
                   - delta_phi_e 
                   + eta_e 
                   +  delta_phi_s
                    - param.R_contact*i_cell*param.A_cc
        )
              

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
            "Electrolyte concentration [mol.m-3]":c_e,
            "Positive particle surface "
            "concentration [mol.m-3]": pybamm.PrimaryBroadcast(
                c_s_surf_p, "positive electrode"
            ),
            "Electrolyte concentration [mol.m-3]": c_e,
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
            pybamm.Event(
                "Minimum electrolyte concentration in negative electrode",
                pybamm.min(c_e_n) - 0.1,
            ),
            pybamm.Event(
                "Minimum electrolyte concentration in positive electrode",
                pybamm.min(c_e_p) - 0.1,
            ),
            pybamm.Event(
                "Minimum negative particle surface stoichiometry",
                pybamm.min(sto_surf_n) - 0.001,
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