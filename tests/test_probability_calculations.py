import pytest
import numpy as np
from pandemic.probability_calculations import (
    probability_of_entry,
    probability_of_establishment,
    probability_of_introduction,
)


def test_probability_of_entry():

    assert (
        probability_of_entry(
            rho_i=1,
            rho_j=0,
            zeta_it=1,
            lamda_c=1,
            T_ijct=500,
            sigma_T=200,
            mu=0.02,
            d_ij=150,
            chi_it=1,
        )
        == 0
    )

    assert (
        probability_of_entry(
            rho_i=0,
            rho_j=1,
            zeta_it=1,
            lamda_c=1,
            T_ijct=500,
            sigma_T=200,
            mu=0.02,
            d_ij=150,
            chi_it=1,
        )
        == 0
    )

    assert (
        probability_of_entry(
            rho_i=0,
            rho_j=0,
            zeta_it=0,
            lamda_c=1,
            T_ijct=500,
            sigma_T=200,
            mu=0.02,
            d_ij=150,
            chi_it=1,
        )
        == 0
    )

    assert (
        probability_of_entry(
            rho_i=0,
            rho_j=0,
            zeta_it=1,
            lamda_c=1,
            T_ijct=500,
            sigma_T=200,
            mu=0.02,
            d_ij=150,
            chi_it=0,
        )
        == 0
    )

    # assert (
    #     probability_of_entry(
    #         rho_i=0,
    #         rho_j=0,
    #         zeta_it=1,
    #         lamda_c=1,
    #         T_ijct=200,
    #         sigma_T=200,
    #         mu=0.0002,
    #         d_ij=2000,
    #         chi_it=1,
    #     )
    #     == 0
    # )


def test_probability_of_establishment():
    assert (
        probability_of_establishment(
            alpha=0,
            beta=1,
            delta_kappa_ijt=1,
            sigma_kappa=1,
            h_jt=1,
            sigma_h=1,
            epsilon_jt=1,
            sigma_epsilon=1,
            phi=1,
            sigma_phi=2,
        )
        == 0
    )

    assert (
        0.0
        <= probability_of_establishment(
            alpha=0.4,
            beta=1,
            delta_kappa_ijt=1,
            sigma_kappa=1,
            h_jt=1,
            sigma_h=1,
            epsilon_jt=1,
            sigma_epsilon=1,
            phi=1,
            sigma_phi=2,
        )
        <= 1.0
    )

    assert (
        0.0
        <= probability_of_establishment(
            alpha=0.4,
            beta=1,
            delta_kappa_ijt=0.5,
            sigma_kappa=0.5,
            h_jt=1,
            sigma_h=1,
            epsilon_jt=1,
            sigma_epsilon=1,
            phi=5,
            sigma_phi=2,
        )
        <= 1.0
    )


def test_probability_of_introduction():
    assert (
        probability_of_introduction(
            probability_of_entry_ijct=0, probability_of_establishment_ijt=1
        )
        == 0
    )

    assert (
        probability_of_introduction(
            probability_of_entry_ijct=1, probability_of_establishment_ijt=1
        )
        == 1
    )
