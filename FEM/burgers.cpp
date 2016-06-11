#include "burgers.h"
namespace Burgers{

SolidBurgers::SolidBurgers(const Matrix* data)
{
    GK0 = (*data)(0); //Kelvin shear modulus
    mK = (*data)(1); //dependency parameter for "
    etaK0 = (*data)(2); //Kelvin viscosity
    mvK = (*data)(3); //dependency parameter for "
    GM0 = (*data)(4); //Maxwell shear modulus
    KM0 = (*data)(5); //Maxwell bulk modulus
    etaM0 = (*data)(6); //Maxwell viscosity
    mvM = (*data)(7); //dependency parameter for "
    m_GM = (*data)(8); // slope of elesticity temperature dependence
    m_KM = (*data)(9); // slope of elesticity temperature dependence
    T_ref = (*data)(10); // reference temperature dependency parameter for "
    B = (*data)(11); // constant factor for Arrhenius term
    Q = (*data)(12); // activation energy in Arrhenius term

    GM = GM0;
    KM = KM0;
    GK = GK0;
    etaK = etaK0;
    etaM = etaM0;

    smath = new Invariants();
}

SolidBurgers::~SolidBurgers()
{
    smath = NULL;
}

/**************************************************************************
   FEMLib-Method: Burgers::UpdateBurgersProperties()
   Task: Updates BURGERS material parameters in LUBBY2 fashion
   Programing:
   07/2014 TN Implementation
**************************************************************************/
void SolidBurgers::UpdateBurgersProperties(double s_eff, const double Temperature)
{
	double dT = Temperature - T_ref;
	GM = GM0 + m_GM*dT;
	KM = KM0 + m_KM*dT;
	GK = GK0 * std::exp(mK * s_eff);
	etaK = etaK0 * std::exp(mvK * s_eff);
    etaM = etaM0 * std::exp(mvM * s_eff) * B * std::exp(Q/(8.314472*Temperature));
}

/**************************************************************************
   FEMLib-Method: Burgers::CalResidualBurgers()
   Task: Calculates the 12x1 residual vector. Implementation fully implicit only.
   Programing:
   06/2014 TN Implementation
**************************************************************************/
void SolidBurgers::CalResidualBurgers(const double dt, const Eigen::Matrix<double,6,1> &strain_curr, const Eigen::Matrix<double,6,1> &stress_curr,
                                      Eigen::Matrix<double,6,1> &strain_Kel_curr, const Eigen::Matrix<double,6,1> &strain_Kel_t,
                                      Eigen::Matrix<double,6,1> &strain_Max_curr, const Eigen::Matrix<double,6,1> &strain_Max_t, Eigen::Matrix<double,18,1> &res)
{
    Eigen::Matrix<double,6,1> G_j;

    //calculate stress residual
    G_j = stress_curr - 2. * (strain_curr - strain_Kel_curr - strain_Max_curr);
    res.block<6,1>(0,0) = G_j;

    //calculate Kelvin strain residual
    G_j = 1./dt * (strain_Kel_curr - strain_Kel_t) - 1./(2.*etaK) * (GM*stress_curr - 2.*GK*strain_Kel_curr);
    res.block<6,1>(6,0) = G_j;

    //calculate Maxwell strain residual
    G_j = 1./dt * (strain_Max_curr - strain_Max_t) - GM/(2.*etaM)*stress_curr;
    res.block<6,1>(12,0) = G_j;


}

/**************************************************************************
   FEMLib-Method: Burgers::CalJacobianBurgers()
   Task: Calculates the 12x12 Jacobian. Implementation fully implicit only.
   Programing:
   06/2014 TN Implementation
**************************************************************************/
void SolidBurgers::CalJacobianBurgers(const double dt, Eigen::Matrix<double,18,18> &Jac, double s_eff, const Eigen::Matrix<double,6,1> &sig_i, const Eigen::Matrix<double,6,1> &eps_K_i)
{
    //6x6 submatrices of the Jacobian
    Eigen::Matrix<double,6,6> G_ij;
    //avoid division by 0
    s_eff = std::max(s_eff,DBL_EPSILON);

    //aid terms
    Eigen::Matrix<double,6,1> eps_K_aid, dG_K, dmu_vK, dmu_vM;

    eps_K_aid = 1./(etaK*etaK)*(GM*sig_i-2.*GK*eps_K_i);
    dG_K = mK * 3.*GK*GM/(2.*s_eff)*sig_i;
    dmu_vK = mvK * 3.*GM*etaK/(2.*s_eff)*sig_i;
    dmu_vM = mvM * 3.*GM*etaM/(2.*s_eff)*sig_i;

    //Check Dimension of Jacobian
    if (Jac.cols() != 18 || Jac.rows() != 18)
    {
        std::cout << "WARNING: Jacobian given to Burgers::CalJacobianBurgers has wrong size. Resizing to 18x18\n";
        Jac.resize(18,18);
    }
    Jac.setZero(18,18);

    //build G_11
    G_ij = smath->ident;
    Jac.block<6,6>(0,0) = G_ij;

    //build G_12
    G_ij = 2. * smath->ident;
    Jac.block<6,6>(0,6) = G_ij;

    //build G_13
    G_ij = 2. * smath->ident;
    Jac.block<6,6>(0,12) = G_ij;

    //build G_21
    G_ij = -GM/(2.*etaK) * smath->ident + 0.5*eps_K_aid*dmu_vK.transpose() + 1./etaK * eps_K_i * dG_K.transpose();
    Jac.block<6,6>(6,0) = G_ij;

    //build G_22
    G_ij = (1./dt + GK/etaK) * smath->ident;
    Jac.block<6,6>(6,6) = G_ij;

    //nothing to do for G_23

    //build G_31
    G_ij = - GM/(2.*etaM)* smath->ident + GM/(2.*etaM*etaM) * sig_i * dmu_vM.transpose();
    Jac.block<6,6>(12,0) = G_ij;

    //nothing to do for G_32

    //build G_33
    G_ij = 1./dt * smath->ident;
    Jac.block<6,6>(12,12) = G_ij;
}

/**************************************************************************
   FEMLib-Method: Burgers::CaldGdEBurgers()
   Task: Calculates the 12x6 derivative of the residuals with respect to total strain. Implementation fully implicit only.
   Programing:
   06/2014 TN Implementation
**************************************************************************/
void SolidBurgers::CaldGdEBurgers(Eigen::Matrix<double,18,6> &dGdE)
{

    //6x6 submatrices of the Jacobian
    //Eigen::Matrix<double, 6, 6> dGdE_1(1./dt *D_Max);
    Eigen::Matrix<double, 6, 6> dGdE_1;

    dGdE_1 = -2. * smath->ident;

    //Check Dimension of dGdE
    if (dGdE.cols() != 6 || dGdE.rows() != 18)
    {
        std::cout << "WARNING: dGdE given to Burgers::CaldGdEBurgers has wrong size. Resizing to 18x6\n";
        dGdE.resize(18,6);
    }

    dGdE.setZero(18,6);
    dGdE.block<6,6>(0,0) = dGdE_1;

}

}
