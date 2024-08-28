#
# Bulter volmer class
#

import pybamm
from .base_kinetics import BaseKinetics


class SymmetricButlerVolmer(BaseKinetics):
    """
    Submodel which implements the symmetric forward Butler-Volmer equation:

    .. math::
        j = 2 * j_0(c) * \\sinh(ne * F * \\eta_r(c) / RT)

    Parameters
    ----------
    param : parameter class
        model parameters
    domain : str
        The domain to implement the model, either: 'Negative' or 'Positive'.
    reaction : str
        The name of the reaction being implemented
    options: dict
        A dictionary of options to be passed to the model.
        See :class:`pybamm.BaseBatteryModel`
    phase : str, optional
        Phase of the particle (default is "primary")
    """

    def __init__(self, param, domain, reaction, options, phase="primary"):
        super().__init__(param, domain, reaction, options, phase)

    def _get_kinetics(self, j0, ne, eta_r, T, u):
        Feta_RT = self.param.F * eta_r / (self.param.R * T)
        return 2 * u * j0 * pybamm.sinh(ne * 0.5 * Feta_RT)


class AsymmetricButlerVolmer(BaseKinetics):
    """
    Submodel which implements the asymmetric forward Butler-Volmer equation

    Parameters
    ----------
    param : parameter class
        model parameters
    domain : str
        The domain to implement the model, either: 'Negative' or 'Positive'.
    reaction : str
        The name of the reaction being implemented
    options: dict
        A dictionary of options to be passed to the model.
        See :class:`pybamm.BaseBatteryModel`
    phase : str, optional
        Phase of the particle (default is "primary")
    """

    def __init__(self, param, domain, reaction, options, phase="primary"):
        super().__init__(param, domain, reaction, options, phase)

    def _get_kinetics(self, j0, ne, eta_r, T, u):
        alpha = self.phase_param.alpha_bv
        Feta_RT = self.param.F * eta_r / (self.param.R * T)
        arg_ox = ne * alpha * Feta_RT
        arg_red = -ne * (1 - alpha) * Feta_RT
        return u * j0 * (pybamm.exp(arg_ox) - pybamm.exp(arg_red))
    

class kineticsControlledButlerVolmer(BaseKinetics):
    
    """
    Submodel which implements the asymmetric kinetic controlled  forward Butler-Volmer equation
    .. math::
        j = j_0(c) * \\((c_s_surf/c_avg)*exp(ne * F * \\eta_r(c) / RT) 
        - ((c_max - c_surf)/(c_max - c_avg)) * (c_e/c_e_avg) * exp(-ne * F * \\eta_r(c) / RT))

    Parameters
    ----------
    param : parameter class
        model parameters
    domain : str
        The domain to implement the model, either: 'Negative' or 'Positive'.
    reaction : str
        The name of the reaction being implemented
    options: dict
        A dictionary of options to be passed to the model.
        See :class:`pybamm.BaseBatteryModel`
    phase : str, optional
        Phase of the particle (default is "primary")
    """

    def __init__(self, param, domain, reaction, options, phase="primary"):
        super().__init__(param, domain, reaction, options, phase)

    def _get_kinetics(self, j0, ne, eta_r, T, u,
                       c_rt, c_diff_rt, c_rt_e):
        alpha = self.phase_param.alpha_bv
        Feta_RT = self.param.F * eta_r / (self.param.R * T)
        arg_ox = ne * alpha * Feta_RT
        arg_red = -ne * (1 - alpha) * Feta_RT
        return u * j0 * (c_rt*pybamm.exp(arg_ox) - c_diff_rt*c_rt_e*pybamm.exp(arg_red))
