import pybamm
import numpy as np
from pybamm import Interpolant, exp


def hard_carbon_diffusivity_Chayambuka2022(sto, T):
    """
    Hard Carbon diffusivity as a function of stochiometry.
    A polynomial fit is  performed for the values extracted from  Fig A1 
    of  Chayambuka [1].

    The activition energy for hard carbon is taken from Bowen [2]. (Eq 6) 

    References
    ----------
    .. [1] https://doi.org/10.1016/j.electacta.2021.139764
    .. [2] https://eprints.soton.ac.uk/470056/2/batteries_08_00108_v2.pdf
   
    Parameters
    ----------
    sto: :class:`pybamm.Symbol`
        Electrode stochiometry
    T: :class:`pybamm.Symbol`
        Dimensional temperature

    Returns
    -------
    :class:`pybamm.Symbol`
        Solid diffusivity
    """
    D_ref = ( 
        -4.6741e-13*sto**6 + 1.3106e-12*sto**5 
        - 1.3034e-12*sto**4 + 5.4204e-13*sto**3 
        - 8.865e-14*sto**2 + 7.9521e-15*sto**1 
        + 1.5172e-16  
    ) 
    
    # m**2/s fitting optimization
    # D_ref = 1.25 * D_ref

    # fitti

    # D_ref = 6e-16
    # D_ref = 2e-15
 
  
    E_D_s = 16130 # J/mol
    arrhenius = exp(E_D_s / pybamm.constants.R * (1 / 298.15 - 1 / T))

    return D_ref * arrhenius


def hard_carbon_ocp_Chayambuka2022(sto):
    """
    Hard Carbon Open-circuit Potential (OCP) as a function of the
    stochiometry. The fit is performed for the values extracted from emf curves in 
    Fig 4c of Chayambuka [1].
    The full capcity of the cell  HC//NVPF  is taken as  2.81 mAh based on 
    Fig 12 Chayambuka [2].
    The fit expressions are available in Garaoati et al., [3]

    References
    ----------
    .. [1] https://doi.org/10.1016/j.electacta.2021.139764
    .. [2] https://doi.org/10.1016/j.electacta.2021.139726
    .. [3] https://dx.doi.org/10.1149/1945-7111/acb01b
    """
    # # based on Garapati et al., 
    # u_eq = (2.157 - 9.947 * sto + 18.164 * sto**2
    #          - 14.672 * sto**3 + 4.473 * sto**4
    # )

    # fit assuming sto = transferred charge/Qmax where Qmax = 2.81 
        
    # u_eq = (-542.089 * sto**7 
    #          + 2142.85 * sto**6 
    #          - 3450.78 * sto**5 
    #          + 2908.78 * sto**4
    #          - 1375.49 * sto**3
    #          + 366.476 * sto**2 
    #          - 53.9669 * sto
    #          + 4.22611)
    # fit assuming that Q = soc * () instead of (soc_max - soc)* ()  
    # # -- need to try
    # latest fit assuming  Q = (soc_max - soc)* ()  
 
    # u_eq = (1856.4 * sto**8
    #         -8475.44 * sto**7 
    #         + 16255.9 * sto**6 
    #         - 16989.6 * sto**5 
    #         + 10492.1 * sto**4
    #         - 3883.79 * sto**3
    #         + 838.052 * sto**2 
    #         - 99.3537 * sto
    #         + 5.92005)
    #
    # comosol data and fit
    # sto_ref =  np.array([0.001436794, 0.001643334, 0.00811789, 
    #             0.01027308, 0.035479844, 0.060758448, 
    #             0.090185795, 0.127982471, 0.169900951,
    #             0.232688871, 0.320485995, 0.403954777, 
    #             0.483167055, 0.574861484, 0.687380455, 
    #             0.799899424, 0.891593855, 0.960353451, 
    #             0.987446008, 0.995806356])
    
    # un_ref = np.array([1.318963892, 1.21982507, 1.112038542, 
    #           1.077546494, 0.978299913, 0.844570264,
    #             0.719443422, 0.577039126, 0.456168787, 
    #             0.317967115, 0.1753473, 0.110332349, 0.088439192, 
    #             0.075112923, 0.066007238, 0.056901553,
    #             0.043575284, 0.038968561, 0.034541438, 0.021574368])

    # u_eq = Interpolant(sto_ref, un_ref, sto, interpolator="cubic")

    p1, p2, p3, p4, p5, p6, p7, p8, p9, p10 = [
        0.01382, -11.93412, 250.07768, -407.84311, 
        339.82134, 0.01265, -10.55755, 
        184.91027, 175.59292, 4033.17342]
    x = sto 
    u_eq =  ((p1 +  p2*x + p3 * x**2 
              + p4 * x**3 + p5 * x**4)/
            (p6 + p7*x + p8*x**2 
             + p9*x**3 + p10*x**4)
    )

    return u_eq

def _kn(c_s_surf, T):
    """
    kinetic rate constant of negative electrode for Intercalation reaction of Na.
    To be used in computing the Excahnge current density for Butler-Volmer reactions

    Polynomial fit to the data extracted from Chayambuka 2022 [2].

    References
    --------------
    .. [1] https://doi.org/10.1016/j.electacta.2021.139764

    Parameters
    -------------
    c_s_surf : :class:`pybamm.Symbol`
        Particle concentration [mol.m-3]
    T : :class:`pybamm.Symbol`
        Temperature [K]

    Returns
    -------
    :class:`pybamm.Symbol`
        kinetic Rate  constant  [m2.5.mol-0.5.s-1]
    """
    kn_val = (-1.4132e-29 * c_s_surf**5 +  5.5715e-25 * c_s_surf**4 
              -7.5952e-21 * c_s_surf**3 +  4.0981e-17 * c_s_surf**2 
              -6.9205e-14 * c_s_surf +  4.1515e-11)
              
    # # adding a factor of 0.5 to optimize the fit for the P2D model
    # kn_val = 0.55 * kn_val 

    # # fit from GITT data
    # kn_val = (
    #     -2.2888e-30 * c_s_surf**5 + 8.9841e-26 * c_s_surf**4
    #     -1.2176e-21 * c_s_surf**3 + 6.5076e-18 * c_s_surf**2
    #     -1.0725e-14 * c_s_surf + 6.2747e-12
    # )
    return kn_val


def hard_carbon_electrolyte_exchange_current_density_Chayambuka2022(
    c_e, c_s_surf,  c_s_max, T
):
# def hard_carbon_electrolyte_exchange_current_density_Chayambuka2022(
#     c_e_avg, c_s_surf, c_s_ravg, c_s_max, T
# ):
    """
    Exchange-current density for Butler-Volmer reactions between hard Carbon and NVPF in EC:PC.[1] Chayambuka 2022

    ** Arrhenius value left same as the graphite value from dualfoil needs correction 

    Notable differences from Lithium :
    .. 1) Average intercalated Na+ value instaed of surf and average electrolyte concentration in evaluation of j0 
    .. 2) kinetic rate constant as a function of concentration
         (new in pybamm as most of the paramater sets just have an constant value) 


    References
    ----------
    .. [1] https://doi.org/10.1016/j.electacta.2021.139764

    Parameters
    ----------
    c_e_avg : :class:`pybamm.Symbol`
        Average Electrolyte concentration across x [mol.m-3]
    c_s_surf : :class:`pybamm.Symbol`
        Particle concentration [mol.m-3]
    c_s_ravg : :class:`pybamm.Symbol`
        Particle average concentration across 
        r domain [mol.m-3]
    c_s_max : :class:`pybamm.Symbol`
        Maximum particle concentration [mol.m-3]
    T : :class:`pybamm.Symbol`
        Temperature [K]

    Returns
    -------
    :class:`pybamm.Symbol`
        Exchange-current density [A.m-2]
    """
    # m_ref = 7e-11 * pybamm.constants.F  #   (A/m2)(m3/mol)**1.5 
    # m_ref = 1e-12 * pybamm.constants.F  #   (A/m2)(m3/mol)**1.5  
    # m_ref = 6e-12 * pybamm.constants.F  #   (A/m2)(m3/mol)**1.5   # best fit for dfn 
    # m_ref = 2e-12 * pybamm.constants.F  #   (A/m2)(m3/mol)**1.5 
    m_ref = _kn(c_s_surf, T) * pybamm.constants.F  #   (A/m2)(m3/mol)**1.5 
    E_r = 37480
    arrhenius = exp(E_r / pybamm.constants.R * (1 / 298.15 - 1 / T))
    
    return m_ref * arrhenius * c_e**0.5 * c_s_surf**0.5 * (c_s_max - c_s_surf) ** 0.5
    # return m_ref * arrhenius * c_e_avg**0.5 * c_s_ravg**0.5 * (c_s_max - c_s_ravg) ** 0.5


def hard_carbon_entropic_change_Chayambuka2022(sto, c_s_max):
    """
    Hard Carbon entropic change in open-circuit potential (OCP) at a temperature of
    298.15K 
    No data found as of now, so taking it as zero
    
    Parameters
    ----------
    sto : :class:`pybamm.Symbol`
        Stochiometry of material (li-fraction)

    """
    du_dT = 0

    return du_dT


def NVPF_diffusivity_Chayambuka2022(sto, T):
    """
    NVPF diffusivity as a function of stochiometry.
    A polynomial fit is  performed for the values extracted from  Fig A2 
    of  Chayambuka [1].

    Couldn't find ay records for the arrhenius activity coefficient for NVPF keeping it same value as LiCO2 from Dual foil [2]

    References
    ----------
    .. [1] https://doi.org/10.1016/j.electacta.2021.139764
    .. [2] http://www.cchem.berkeley.edu/jsngrp/fortran.html


    Parameters
    ----------
    sto: :class:`pybamm.Symbol`
        Electrode stochiometry
    T: :class:`pybamm.Symbol`
        Dimensional temperature

    Returns
    -------
    :class:`pybamm.Symbol`
        Solid diffusivity
    """
    D_ref = (7.1008e-14 * sto**6 - 2.8245e-13 * sto**5
             + 4.5005e-13 * sto**4 - 3.6495e-13 * sto**3
             + 1.5745e-13 * sto**2 - 3.4015e-14 * sto
             + 2.9309e-15)
    
    # fitting ignoring the data points nearto soc ~ 100 
    # D_ref = (   5.878e-14 * sto**8 - 2.6e-13 * sto**7
    #             + 5.442e-13 * sto**6 - 7.388e-13 * sto**5
    #             + 7.003e-13 * sto**4 - 4.426e-13 * sto**3
    #             + 1.702e-13 * sto**2 - 3.496e-14 * sto
    #             + 2.952e-15
    # )  # m**2/s

    E_D_s = 18550    # J/mol
    arrhenius = exp(E_D_s / pybamm.constants.R * (1 / 298.15 - 1 / T))

    return D_ref * arrhenius



def NVPF_ocp_Chayambuka2022(sto):
    """
     Open-circuit Potential (OCP) as a a function of the
    stochiometry. The fit is performed for the emf values 
    extracted from  plot Fig 4b of Chayambuka [1]. 
    The fit expressions are available in Garapati et al., [2]
    References
    ----------
    .. [1] https://doi.org/10.1016/j.electacta.2021.139764
    .. [2] https://dx.doi.org/10.1149/1945-7111/acb01b

    Parameters
    ----------
    sto : :class:`pybamm.Symbol`
       Stochiometry of material (li-fraction)

    """
   
    # x = sto
  
    # p1, p2, p3, p4, p5, p6, p7, p8, p9, p10 =  [  
    #     116.1647,  -940.2551,  3101.9987,
    #     -5292.6225,  4040.4367,    27.9167,
    #     -228.1831,   769.0709, -1353.4191,  
    #     1063.2453
    # ]

    # u_eq = ((p1 + p2*x**2 + p3*x**4  + p4*x**6 + p5*x**8)/
    #         (p6 + p7*x**2 + p8*x**4 + p9*x**6 + p10*x**8)
    # )

    # # data from comsol 
    sto_ref = np.array([0.21, 0.21004478, 0.219269532, 0.261497402, 
               0.332354842, 0.409317336, 0.474174208, 
               0.526985168, 0.560316674, 0.575706188, 
               0.582273973, 0.585975815, 0.590901654,
               0.598588947, 0.655863013, 0.696598205, 
               0.741871138, 0.802205196, 0.864031932, 
               0.930500897, 0.959279735, 0.979341333, 
               0.980053837, 0.983058101, 0.992282853, 
               0.996268304, 0.999940293])
    
    up_ref = np.array([4.288102031, 4.210892773, 4.175283892, 
              4.172472367, 4.172738781, 4.158176665, 
              4.152479924, 4.143767596, 4.11121965, 
              4.048901275, 3.941995276, 3.805375531,
              3.725196032, 3.695521965, 3.698707605, 
              3.69292017, 3.684179499, 3.678465753, 
              3.675727917, 3.649245158, 3.62262069, 
              3.530616911, 3.450437411, 3.391026925,
              3.355418044, 3.162363722, 3.031684459])
    

    u_eq = Interpolant(sto_ref, up_ref, sto, interpolator="linear")
    return u_eq

def _kp(c_s_surf, T):
    """
    kinetic rate constant of psitive electrode for Intercalation reaction of Na.
    To be used in computing the Excahnge current density for Butler-Volmer reactions

    Polynomial fit to the data extracted from Chayambuka 2022 [2].

    References
    --------------
    .. [1] https://doi.org/10.1016/j.electacta.2021.139764

    Parameters
    -------------
    c_s_surf : :class:`pybamm.Symbol`
        Particle concentration [mol.m-3]
    T : :class:`pybamm.Symbol`
        Temperature [K]

    Returns
    -------
    :class:`pybamm.Symbol`
        kinetic Rate  constant  [m2.5.mol-0.5.s-1]
    """
    kp_val = (4.2884e-34 * c_s_surf**6 - 2.641e-29 * c_s_surf**5 
              + 6.4845e-25 * c_s_surf**4 - 8.06e-21 * c_s_surf**3 
              + 5.3039e-17 * c_s_surf**2 - 1.7493e-13 * c_s_surf
              + 2.3728e-10)
    return kp_val
    
  
def NVPF_electrolyte_exchange_current_density_Chayambuka2022(c_e, c_s_surf,  c_s_max, T):
# def NVPF_electrolyte_exchange_current_density_Chayambuka2022(c_e_avg, c_s_surf, c_s_ravg, c_s_max, T):
    """
    Exchange-current density for Butler-Volmer reactions between NVPF and NaPF6 in
    EC:PC.[1]

    ** Arrhenius value left same as the LiCoO2 value from dualfoil needs correction 

    Notable differences from Lithium :
    .. 1) Average intercalated Na+ value instaed of surf and average electrolyte concentration in evaluation of j0 
    .. 2) kinetic rate constant as a function of concentration
         (new in pybamm as most of the paramater sets just have an constant value) 

    References
    ----------
    .. [1] https://doi.org/10.1016/j.electacta.2021.139764

    Parameters
    ----------
    c_e_avg : :class:`pybamm.Symbol`
        Average Electrolyte concentration across x [mol.m-3]
    c_s_surf : :class:`pybamm.Symbol`
        Particle concentration [mol.m-3]
    c_s_ravg : :class:`pybamm.Symbol`
        Particle average concentration across 
        r domain [mol.m-3]
    c_s_max : :class:`pybamm.Symbol`
        Maximum particle concentration [mol.m-3]
    T : :class:`pybamm.Symbol`
        Temperature [K]

    Returns
    -------
    :class:`pybamm.Symbol`
        Exchange-current density [A.m-2]
    """ 
    # m_ref = 3e-11 *  pybamm.constants.F # (A/m2)(m3/mol)**1.5 
    m_ref = _kp(c_s_surf, T) *  pybamm.constants.F # (A/m2)(m3/mol)**1.5 
    E_r = 39570
    arrhenius = exp(E_r / pybamm.constants.R * (1 / 298.15 - 1 / T))

    return m_ref * arrhenius * c_e**0.5 * c_s_surf**0.5 * (c_s_max - c_s_surf) ** 0.5
    # return m_ref * arrhenius * c_e_avg**0.5 * c_s_ravg**0.5 * (c_s_max - c_s_ravg) ** 0.5


def NVPF_entropic_change_Chayambuka2022(sto, c_s_max):
    """
    Na3V2(PO4)2F3 (NVPF) entropic change in open-circuit potential (OCP) at
    a temperature of 298.15K as a function of the stochiometry. 
    No data available, taking it as zero for the moment.
  

    Parameters
    ----------
    sto : :class:`pybamm.Symbol`
        Stochiometry of material (li-fraction)
    """
    

    du_dT = 0

    return du_dT


def electrolyte_diffusivity_Chayambuka2022(c_e, T):
    """
    Diffusivity of NaPF6 in EC:PC as a function of ion concentration. 
    The P2D model fit data from the Fig A3 of Chayambuka [1]. 
    A polynomial fit is used on the data. 

    Arrhenius value left untouched from dualfoil [2], need to be 
    corrected has no effect in the present work 

    References
    ----------
    .. [1]  https://doi.org/10.1016/j.electacta.2021.139764
    .. [2]  http://www.cchem.berkeley.edu/jsngrp/fortran.html

    Parameters
    ----------
    c_e: :class:`pybamm.Symbol`
        Dimensional electrolyte concentration
    T: :class:`pybamm.Symbol`
        Dimensional temperature


    Returns
    -------
    :class:`pybamm.Symbol`
        Solid diffusivity
    """

    D_c_e = (8.7549e-26 * c_e**4 
              - 1.7006e-21 * c_e**3
              + 9.4034e-18 * c_e**2
              - 2.3662e-14 * c_e 
              + 4.1152e-11
    )
   
    E_D_e = 37040
    arrhenius = exp(E_D_e / pybamm.constants.R * (1 / 298.15 - 1 / T))

    return D_c_e * arrhenius


def electrolyte_conductivity_Chayambuka2022(c_e, T):
    """
    Conductivity of NaPF6 in EC:PC as a function of ion concentration. 
    The P2D model fit data from the Fig A3 of Chayambuka [1]. 
    A polynomial fit is used on the data. 

    Arrhenius value left untouched from dualfoil [2], need to be 
    corrected, has no effect in the present work 

    References
    ----------
    .. [1]  https://doi.org/10.1016/j.electacta.2021.139764
    .. [2]  http://www.cchem.berkeley.edu/jsngrp/fortran.html

    Parameters
    ----------
    c_e: :class:`pybamm.Symbol`
        Dimensional electrolyte concentration
    T: :class:`pybamm.Symbol`
        Dimensional temperature


    Returns
    -------
    :class:`pybamm.Symbol`
        Solid diffusivity
    """

    sigma_e = (
        -6.8163e-14 * c_e**4
        + 5.2618e-10 * c_e**3 
        - 1.5716e-06 * c_e** 2
        + 0.001948 * c_e 
        -0.0017252
    )

    E_k_e = 34700
    arrhenius = exp(E_k_e / pybamm.constants.R * (1 / 298.15 - 1 / T))

    return sigma_e * arrhenius


# Parameters dictionary 

"""
Parameters for a sodium-ion battery (SIB) composed of 
Na3V2(PO4)2F3 (NVPF) and hard carbon (HC) as positive and 
negative electrodes, respectively, from the
paper :footcite:t:`Chayambuka2022`and references therein.
"""

""" total length of cell
L = L_n + L_s + L_p + L_cc_n + L_cc_p 
L = 64 + 68 +25 + 22*2 1e-6 = 201e-6
volume = A_cc *L = 201e-6*2.54 * 1e-4 = 5.1e-8

"""


parameter_values =  pybamm.ParameterValues({
    "chemistry": "sodium_ion",
    
    # cell
    "Negative current collector thickness [m]": 22e-6,
    "Negative electrode thickness [m]": 64e-6,
    "Separator thickness [m]": 2.5e-05,
    "Positive electrode thickness [m]": 68e-6,
    "Positive current collector thickness [m]": 22e-6,
    "Electrode height [m]": 0.0159,
    "Electrode width [m]": 0.0159,
    "Cell volume [m3]": 5.1e-08,
    "Nominal cell capacity [A.h]": 2.9*1e-3,
    "Current function [A]": 8.33*2.5e-4, 
    "Contact resistance [Ohm]": 10.5e-3/2.54e-4,

    # negative electrode
    "Negative electrode conductivity [S.m-1]": 256,
    "Maximum concentration in negative electrode [mol.m-3]": 14540,
    "Negative particle diffusivity [m2.s-1]"
    "": hard_carbon_diffusivity_Chayambuka2022,
    "Negative electrode OCP [V]": hard_carbon_ocp_Chayambuka2022,
    "Negative electrode porosity": 0.51,
    "Negative electrode active material volume fraction": 0.489,
    "Negative particle radius [m]": 3.48e-06,
    "Negative electrode Bruggeman coefficient (electrolyte)": 1.5,
    "Negative electrode Bruggeman coefficient (electrode)": 0,
    "Negative electrode charge transfer coefficient": 0.5,
    "Negative electrode exchange-current density [A.m-2]"
    "": hard_carbon_electrolyte_exchange_current_density_Chayambuka2022,
    "Negative electrode density [kg.m-3]": 1950.0,
    "Negative electrode OCP entropic change [V.K-1]"
    "": hard_carbon_entropic_change_Chayambuka2022,

    # positive electrode
    "Positive electrode conductivity [S.m-1]": 50.0,
    "Maximum concentration in positive electrode [mol.m-3]": 15320,
    "Positive particle diffusivity [m2.s-1]": NVPF_diffusivity_Chayambuka2022,
    "Positive electrode OCP [V]": NVPF_ocp_Chayambuka2022,
    "Positive electrode porosity": 0.23,
    "Positive electrode active material volume fraction": 0.55,
    "Positive particle radius [m]": 0.59e-06,
    "Positive electrode Bruggeman coefficient (electrolyte)": 1.5,
    "Positive electrode Bruggeman coefficient (electrode)": 0,
    "Positive electrode charge transfer coefficient": 0.5,
    "Positive electrode exchange-current density [A.m-2]"
    "": NVPF_electrolyte_exchange_current_density_Chayambuka2022,
    "Positive electrode density [kg.m-3]": 3200.0,
    "Positive electrode OCP entropic change [V.K-1]"
    "": NVPF_entropic_change_Chayambuka2022,

    # separator
    "Separator porosity": 0.55,
    "Separator Bruggeman coefficient (electrolyte)": 1.5,
    "Separator density [kg.m-3]": 1400.0,
   
    # electrolyte
    "Initial concentration in electrolyte [mol.m-3]": 1000.0,
    "Cation transference number": 0.45,
    "Thermodynamic factor": 1.0,
    "Electrolyte diffusivity [m2.s-1]": electrolyte_diffusivity_Chayambuka2022,
    "Electrolyte conductivity [S.m-1]": electrolyte_conductivity_Chayambuka2022,

    # experiment
    "Reference temperature [K]": 298.15,
    "Ambient temperature [K]": 298.15,
    "Number of electrodes connected in parallel to make a cell": 1.0,
    "Number of cells connected in series to make a battery": 1.0,
    "Lower voltage cut-off [V]": 2,
    "Upper voltage cut-off [V]": 4.2,
    "Open-circuit voltage at 0% SOC [V]": 3,
    "Open-circuit voltage at 100% SOC [V]": 4.3,
    "Initial concentration in negative electrode [mol.m-3]": 14520,
    "Initial concentration in positive electrode [mol.m-3]": 3320,
    "Initial temperature [K]": 298.15,
    # citations
    "citations": ["Chayambuka2022"],
} )


"""thermal and other additional parameters  might be needed for adding 
in while performing thermal modelling

additional parameters set = {
 # "Negative tab width [m]": 0.04,
    # "Negative tab centre y-coordinate [m]": 0.06,
    # "Negative tab centre z-coordinate [m]": 0.137,
    # "Positive tab width [m]": 0.04,
    # "Positive tab centre y-coordinate [m]": 0.147,
    # "Positive tab centre z-coordinate [m]": 0.137,
    # "Negative electrode double-layer capacity [F.m-2]": 0.2,
"Cell cooling surface area [m2]": 0.0569,
"Negative current collector conductivity [S.m-1]": 59600000.0,
"Positive current collector conductivity [S.m-1]": 35500000.0,
"Negative current collector density [kg.m-3]": 8954.0,
"Positive current collector density [kg.m-3]": 2707.0,
"Negative current collector specific heat capacity [J.kg-1.K-1]": 385.0,
"Positive current collector specific heat capacity [J.kg-1.K-1]": 897.0,
"Negative current collector thermal conductivity [W.m-1.K-1]": 401.0,
"Separator specific heat capacity [J.kg-1.K-1]": 700.0,
"Separator thermal conductivity [W.m-1.K-1]": 0.16,
"Positive current collector thermal conductivity [W.m-1.K-1]": 237.0,
"Negative electrode specific heat capacity [J.kg-1.K-1]": 700.0,
"Negative electrode thermal conductivity [W.m-1.K-1]": 1.7,
"Negative current collector surface heat transfer coefficient [W.m-2.K-1]"
"": 0.0,
"Positive current collector surface heat transfer coefficient [W.m-2.K-1]"
"": 0.0,
"Positive electrode double-layer capacity [F.m-2]": 0.2,
"Positive electrode specific heat capacity [J.kg-1.K-1]": 700.0,
"Positive electrode thermal conductivity [W.m-1.K-1]": 2.1,
"Negative tab heat transfer coefficient [W.m-2.K-1]": 10.0,
"Positive tab heat transfer coefficient [W.m-2.K-1]": 10.0,
"Edge heat transfer coefficient [W.m-2.K-1]": 0.3,
"Total heat transfer coefficient [W.m-2.K-1]": 10.0,
}


"""
