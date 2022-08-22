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
            min_Tc=0,
            max_Tc=10000,
            mu=0.02,
            d_ij=150,
            chi_it=1,
            lamda_c_weight=0.2,
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
            min_Tc=0,
            max_Tc=10000,
            mu=0.02,
            d_ij=150,
            chi_it=1,
            lamda_c_weight=0,
        )
        == 0
    )

    assert (
        probability_of_entry(
            rho_i=1,
            rho_j=1,
            zeta_it=0,
            lamda_c=1,
            T_ijct=500,
            min_Tc=0,
            max_Tc=10000,
            mu=0.02,
            d_ij=150,
            chi_it=1,
            lamda_c_weight=0.5,
        )
        == 0
    )

    assert (
        probability_of_entry(
            rho_i=1,
            rho_j=1,
            zeta_it=1,
            lamda_c=1,
            T_ijct=500,
            min_Tc=0,
            max_Tc=10000,
            mu=0.02,
            d_ij=150,
            chi_it=0,
            lamda_c_weight=0.2,
        )
        == 0
    )

    assert (
        0
        < probability_of_entry(
            rho_i=0,
            rho_j=0.5,
            zeta_it=1,
            lamda_c=1,
            T_ijct=500,
            min_Tc=0,
            max_Tc=10000,
            mu=0.02,
            d_ij=150,
            chi_it=1,
            lamda_c_weight=0.2,
        )
        <= 1
    )


def test_probability_of_establishment():
    assert (
        probability_of_establishment(
            alpha=0,
            beta=1,
            delta_kappa_ijt=1,
            sigma_kappa=1,
            h_jt=1,
            sigma_h=1,
            phi=1,
            w_phi=2,
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
            phi=1,
            w_phi=2,
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
            phi=5,
            w_phi=2,
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

    assert (
        0
        <= probability_of_introduction(
            probability_of_entry_ijct=0.5, probability_of_establishment_ijt=0.25
        )
        <= 1
    )
