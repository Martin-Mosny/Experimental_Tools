/* GRChombo
 * Copyright 2012 The GRChombo collaboration.
 * Please refer to LICENSE in GRChombo's root directory.
 */

#include "SetLevelData.H"
#include "AMRIO.H"
#include "BCFunc.H"
#include "BRMeshRefine.H"
#include "BiCGStabSolver.H"
#include "BoxIterator.H"
#include "CONSTANTS.H"
#include "CoarseAverage.H"
#include "LoadBalance.H"
#include "MyPhiFunction.H"
#include "MyPiFunction.H"
#include "PoissonParameters.H"
#include "SetBinaryBH.H"
#include "VariableCoeffPoissonOperatorFactory.H"
#include "computeNorm.H"
#include "parstream.H"
#include <cmath>
#include <random>
#include <chrono>
#include <vector>

// Set various LevelData functions across the grid

// This takes an IntVect and writes the physical coordinates to a RealVect
inline void get_loc(RealVect &a_out_loc, const IntVect &a_iv,
                    const RealVect &a_dx, const PoissonParameters &a_params)
{
    a_out_loc = a_iv + 0.5 * RealVect::Unit;
    a_out_loc *= a_dx;
    a_out_loc -= a_params.domainLength / 2.0;
}

// Create the perturbations in pi to be a Gaussian random field.
void set_Gaussian_pi(Vector<LevelData<FArrayBox> *> &multigrid_vars,
                    const PoissonParameters &a_params)
{
    // Initialize FFTW objects and create the in and out arrays along with the plan
    fftw_complex *in;
    double *out;
    fftw_plan plan;
    int n1 = a_params.nCells[0] * pow(2, a_params.maxLevel);
    int n2 = a_params.nCells[1] * pow(2, a_params.maxLevel);
    int n3 = a_params.nCells[2] * pow(2, a_params.maxLevel);
    in = (fftw_complex*) fftw_malloc(n1 * n2 * (n3 / 2 + 1) * sizeof(fftw_complex));
    out = (double*) fftw_malloc( n1 * n2 * n3 * sizeof(double));
    plan = fftw_plan_dft_c2r_3d(n1, n2, n3, in, out, FFTW_ESTIMATE);

    // Set up the random number generator
    std::default_random_engine generator;
    generator.seed(6);
    std::uniform_real_distribution<double> distribution(0.0, 1.0);

    // Calculate the momentum space random field
    // Declare some variables to be used later
    double phase;
    double amplitude;
    fftw_complex delta;
    int p = 0;
    double q = 0;
    for (int i1 = -n1/2; i1 < n1/2; i1++){
        for (int i2 = -n2/2; i2 < n2/2; i2++){
            for (int i3 = 0; i3 < (n3/2 + 1); i3++){
                
                // Calculate the random Rayleight distribution and phase and assign delta
                // The inverse power of q affects the scale of the Gaussian random field corrrelation
                q = sqrt(pow(i1, 2) + pow(i2, 2) + pow(i3, 2));
                phase = 2 * M_PI * distribution(generator);
                amplitude = sqrt(-log(distribution(generator))/(pow(q, 4) + 0.1) * a_params.Gaussian_amplitude);
                delta[0] = amplitude * cos(phase);
                delta[1] = amplitude * sin(phase);

                // Assign values
                p = i3 + (n3 / 2 + 1) * (((i2) % (n2) + n2) % (n2) + n2* (((i1) % (n1) + n1) % (n1))); 
                in[p][0] = delta[0];
                in[p][1] = delta[1];
            }
        }
    }

    // Now we got to go back and make real the entries that are at points where k = - k mod n
    // For that we define a vector with all the relevant coordinates and another with the relevant q radius 
    std::vector<std::vector<int>> conjugate_vectors({{0, 0, 0}, {n1 / 2, 0 , 0}, {0, n2 / 2, 0}, 
                        {0, 0, n3 / 2}, {n1 / 2, n2 / 2, 0}, {n1 / 2, 0, n3 / 2},
                        {0, n2 / 2, n3 / 2}, {n1 / 2, n2 / 2, n3 / 2}});
    std::vector<double> q_vector({0, n1 / 2.0, n2 / 2.0, n3 / 2.0, sqrt(pow(n1, 2) + pow(n2, 2)) / 2.0, 
                            sqrt(pow(n1, 2) + pow(n3, 2)) / 2.0, sqrt(pow(n2, 2) + pow(n3, 2)) / 2.0,
                            sqrt(pow(n1, 2) + pow(n2, 2) + pow(n3, 2)) / 2.0});

    // Here we then run through these to reassign the real part and set the imaginary part to zero.
    int i;
    for (i = 0; i < 8; i++){
        p = conjugate_vectors[i][2] + n3 * (conjugate_vectors[i][1] + n2 * conjugate_vectors[i][0]);
        in[p][0] = sqrt(-log(distribution(generator)) / (pow(q_vector[i], 3) + 0.1) * a_params.Gaussian_amplitude);
        in[p][1] = 0;
    }
    
    // Run the plan:
    fftw_execute(plan);
    
    int n1_lev;
    int n2_lev;
    int n3_lev;
    // Print the random Gaussian field into the pi field.
    for (int ilev = 0; ilev < a_params.numLevels; ilev++)
    {
        n1_lev = n1 / pow(2, a_params.maxLevel - ilev);
        n2_lev = n2 / pow(2, a_params.maxLevel - ilev);
        n3_lev = n3 / pow(2, a_params.maxLevel - ilev);
        LevelData<FArrayBox> *Fields = multigrid_vars[ilev];

        DataIterator dit = Fields -> dataIterator();
        for (dit.begin(); dit.ok(); ++dit)
        {
            FArrayBox& Fields_box = (*Fields)[dit()];
            Box b = Fields_box.box();
            BoxIterator bit(b);
            for (bit.begin(); bit.ok(); ++bit)
            {
                // Must take the modular coordinate to assign ghost values correctly
                IntVect iv = bit();
                p = (((int)(iv[2] * pow(2, a_params.maxLevel - ilev)) % n3_lev + n3_lev) % n3_lev) 
                  + n3 * ((((int)(iv[1] * pow(2, a_params.maxLevel - ilev)) % n2_lev + n2_lev) % n2_lev) 
                  + n2 * (((int)(iv[0] * pow(2, a_params.maxLevel - ilev)) % n1_lev + n1_lev) % n1_lev));
                Fields_box(iv, c_Pi_0) = a_params.Gaussian_init + out[p];
            }
        }
    }

    fftw_destroy_plan(plan);
    fftw_free(in);
    fftw_free(out);
}




// set initial guess value for the conformal factor psi
// defined by \gamma_ij = \psi^4 \tilde \gamma_ij, scalar field phi
// and \bar Aij = psi^2 A_ij.
// For now the default setup is 2 Bowen York BHs plus a scalar field
// with some initial user specified configuration
void set_initial_conditions(LevelData<FArrayBox> &a_multigrid_vars,
                            LevelData<FArrayBox> &a_dpsi, const RealVect &a_dx,
                            const PoissonParameters &a_params)
{

    CH_assert(a_multigrid_vars.nComp() == NUM_MULTIGRID_VARS);

    DataIterator dit = a_multigrid_vars.dataIterator();
    for (dit.begin(); dit.ok(); ++dit)
    {
        FArrayBox &multigrid_vars_box = a_multigrid_vars[dit()];
        FArrayBox &dpsi_box = a_dpsi[dit()];
        Box b = multigrid_vars_box.box();
        BoxIterator bit(b);
        for (bit.begin(); bit.ok(); ++bit)
        {

            // work out location on the grid
            IntVect iv = bit();

            // set psi to 1.0 and zero dpsi
            // note that we don't include the singular part of psi
            // for the BHs - this is added at the output data stage
            // and when we calculate psi_0 in the rhs etc
            // as it already satisfies Laplacian(psi) = 0
            multigrid_vars_box(iv, c_psi_reg) = 1.0;
            dpsi_box(iv, 0) = 0.0;

            // set the phi value - need the distance from centre
            RealVect loc;
            get_loc(loc, iv, a_dx, a_params);

            // set phi according to user defined function
            multigrid_vars_box(iv, c_phi_0) =
                my_phi_function(loc, a_params.phi_amplitude,
                                a_params.phi_wavelength, a_params.domainLength);
            
            
            // set pi according to use defined function
            #ifndef GAUSSIAN_INIT
            multigrid_vars_box(iv, c_Pi_0) =
                my_pi_function(loc, a_params.phi_amplitude,
                                a_params.phi_wavelength, a_params.domainLength);
            #endif

            // set Aij for spin and momentum according to BH params
            set_binary_bh_Aij(multigrid_vars_box, iv, loc, a_params);
        }
    }
} // end set_initial_conditions



// computes the Laplacian of psi at a point in a box
inline Real get_laplacian_psi(const IntVect &a_iv, const FArrayBox &a_psi_fab,
                              const RealVect &a_dx)
{
    Real laplacian_of_psi = 0.0;
    for (int idir = 0; idir < SpaceDim; ++idir)
    {
        IntVect iv_offset1 = a_iv;
        IntVect iv_offset2 = a_iv;
        iv_offset1[idir] -= 1;
        iv_offset2[idir] += 1;

        // 2nd order stencil for now
        Real d2psi_dxdx = 1.0 / (a_dx[idir] * a_dx[idir]) *
                          (+1.0 * a_psi_fab(iv_offset2) -
                           2.0 * a_psi_fab(a_iv) + 1.0 * a_psi_fab(iv_offset1));
        laplacian_of_psi += d2psi_dxdx;
    }
    return laplacian_of_psi;
} // end get_laplacian_psi

// computes the gradient of the scalar field squared at a point in a box
// i.e. \delta^{ij} d_i phi d_j phi
inline Real get_grad_phi_sq(const IntVect &a_iv, const FArrayBox &a_phi_fab,
                            const RealVect &a_dx)
{
    Real grad_phi_sq = 0.0;
    for (int idir = 0; idir < SpaceDim; ++idir)
    {
        IntVect iv_offset1 = a_iv;
        IntVect iv_offset2 = a_iv;
        iv_offset1[idir] -= 1;
        iv_offset2[idir] += 1;

        // 2nd order stencils for now
        Real dphi_dx =
            0.5 * (a_phi_fab(iv_offset2) - a_phi_fab(iv_offset1)) / a_dx[idir];

        grad_phi_sq += dphi_dx * dphi_dx;
    }
    return grad_phi_sq;
} // end get_grad_phi_sq

// set the rhs source for the poisson eqn
void set_rhs(LevelData<FArrayBox> &a_rhs,
             LevelData<FArrayBox> &a_multigrid_vars, const RealVect &a_dx,
             const PoissonParameters &a_params, const Real constant_K)
{
    CH_assert(a_multigrid_vars.nComp() == NUM_MULTIGRID_VARS);

    DataIterator dit = a_rhs.dataIterator();
    for (dit.begin(); dit.ok(); ++dit)
    {
        FArrayBox &multigrid_vars_box = a_multigrid_vars[dit()];
        FArrayBox &rhs_box = a_rhs[dit()];
        rhs_box.setVal(0.0, 0);
        Box this_box = rhs_box.box(); // no ghost cells

        FArrayBox phi_fab(Interval(c_phi_0, c_phi_0), multigrid_vars_box);
        FArrayBox psi_fab(Interval(c_psi_reg, c_psi_reg), multigrid_vars_box);

        BoxIterator bit(this_box);
        for (bit.begin(); bit.ok(); ++bit)
        {
            IntVect iv = bit();
            RealVect loc;
            get_loc(loc, iv, a_dx, a_params);

            // rhs = m/8 psi_0^5 - 2 pi rho_grad psi_0  - laplacian(psi_0)
            Real m =
                get_m(multigrid_vars_box(iv, c_phi_0), a_params, constant_K);
            Real grad_phi_sq = get_grad_phi_sq(iv, phi_fab, a_dx);
            Real laplacian_of_psi = get_laplacian_psi(iv, psi_fab, a_dx);

            // Also \bar  A_ij \bar A^ij
            Real A2 = 0.0;
            A2 = pow(multigrid_vars_box(iv, c_A11_0), 2.0) +
                 pow(multigrid_vars_box(iv, c_A22_0), 2.0) +
                 pow(multigrid_vars_box(iv, c_A33_0), 2.0) +
                 2 * pow(multigrid_vars_box(iv, c_A12_0), 2.0) +
                 2 * pow(multigrid_vars_box(iv, c_A13_0), 2.0) +
                 2 * pow(multigrid_vars_box(iv, c_A23_0), 2.0);

            Real psi_bl = get_psi_brill_lindquist(loc, a_params);
            Real psi_0 =  multigrid_vars_box(iv, c_psi_reg) + psi_bl;
            Real Pi_field = multigrid_vars_box(iv, c_Pi_0);

            rhs_box(iv, 0) =  0.125 * m * pow(psi_0, 5.0) -
                              //0.125 * A2 * pow(psi_0, -7.0) -
                              M_PI * a_params.G_Newton * grad_phi_sq * psi_0 -
                              M_PI * a_params.G_Newton * pow(Pi_field, 2.0) * pow(psi_0, -1.0) -
                              laplacian_of_psi;
        }
    }
} // end set_rhs

// Set the integrand for the integrability condition for constant K
// when periodic BCs set
void set_constant_K_integrand(LevelData<FArrayBox> &a_integrand,
                              LevelData<FArrayBox> &a_multigrid_vars,
                              const RealVect &a_dx,
                              const PoissonParameters &a_params)
{

    CH_assert(a_multigrid_vars.nComp() == NUM_MULTIGRID_VARS);

    DataIterator dit = a_integrand.dataIterator();
    for (dit.begin(); dit.ok(); ++dit)
    {
        FArrayBox &multigrid_vars_box = a_multigrid_vars[dit()];
        FArrayBox &integrand_box = a_integrand[dit()];
        integrand_box.setVal(0.0, 0);
        Box this_box = integrand_box.box(); // no ghost cells

        FArrayBox phi_fab(Interval(c_phi_0, c_phi_0), multigrid_vars_box);
        FArrayBox psi_fab(Interval(c_psi_reg, c_psi_reg), multigrid_vars_box);

        BoxIterator bit(this_box);
        for (bit.begin(); bit.ok(); ++bit)
        {
            IntVect iv = bit();
            RealVect loc;
            get_loc(loc, iv, a_dx, a_params);

            // integrand = -1.5*m + 1.5 * \bar A_ij \bar A^ij psi_0^-12 +
            // 24 pi rho_grad psi_0^-4  + 12*laplacian(psi_0)*psi^-5
            Real m = get_m(multigrid_vars_box(iv, c_phi_0), a_params, 0.0);
            Real grad_phi_sq = get_grad_phi_sq(iv, phi_fab, a_dx);
            Real laplacian_of_psi = get_laplacian_psi(iv, psi_fab, a_dx);

            // Also \bar  A_ij \bar A^ij
            Real A2 = 0.0;
            A2 = pow(multigrid_vars_box(iv, c_A11_0), 2.0) +
                 pow(multigrid_vars_box(iv, c_A22_0), 2.0) +
                 pow(multigrid_vars_box(iv, c_A33_0), 2.0) +
                 2 * pow(multigrid_vars_box(iv, c_A12_0), 2.0) +
                 2 * pow(multigrid_vars_box(iv, c_A13_0), 2.0) +
                 2 * pow(multigrid_vars_box(iv, c_A23_0), 2.0);

            Real psi_bl = get_psi_brill_lindquist(loc, a_params);
            Real psi_0 = multigrid_vars_box(iv, c_psi_reg) + psi_bl;
            Real Pi_field = multigrid_vars_box(iv, c_Pi_0);

            integrand_box(iv, 0) = -1.5 * m + //1.5 * A2 * pow(psi_0, -12.0) +
                                   12.0 * M_PI * a_params.G_Newton *
                                       grad_phi_sq * pow(psi_0, -4.0) +
                                   12.0 * M_PI * a_params.G_Newton * pow(Pi_field, 2.0) * pow(psi_0, -6) +
                                   12.0 * laplacian_of_psi * pow(psi_0, -5.0);
        }
    }
} // end set_constant_K_integrand

// set the regrid condition - abs value of this drives AMR regrid
void set_regrid_condition(LevelData<FArrayBox> &a_condition,
                          LevelData<FArrayBox> &a_multigrid_vars,
                          const RealVect &a_dx,
                          const PoissonParameters &a_params)
{

    CH_assert(a_multigrid_vars.nComp() == NUM_MULTIGRID_VARS);

    DataIterator dit = a_condition.dataIterator();
    for (dit.begin(); dit.ok(); ++dit)
    {
        FArrayBox &multigrid_vars_box = a_multigrid_vars[dit()];
        FArrayBox &condition_box = a_condition[dit()];
        condition_box.setVal(0.0, 0);
        Box this_box = condition_box.box(); // no ghost cells

        FArrayBox phi_fab(Interval(c_phi_0, c_phi_0), multigrid_vars_box);

        BoxIterator bit(this_box);
        for (bit.begin(); bit.ok(); ++bit)
        {
            IntVect iv = bit();
            RealVect loc;
            get_loc(loc, iv, a_dx, a_params);

            // calculate contributions
            Real m = get_m(multigrid_vars_box(iv, c_phi_0), a_params, 0.0);
            Real grad_phi_sq = get_grad_phi_sq(iv, phi_fab, a_dx);

            // Also \bar  A_ij \bar A^ij
            Real A2 = 0.0;
            A2 = pow(multigrid_vars_box(iv, c_A11_0), 2.0) +
                 pow(multigrid_vars_box(iv, c_A22_0), 2.0) +
                 pow(multigrid_vars_box(iv, c_A33_0), 2.0) +
                 2 * pow(multigrid_vars_box(iv, c_A12_0), 2.0) +
                 2 * pow(multigrid_vars_box(iv, c_A13_0), 2.0) +
                 2 * pow(multigrid_vars_box(iv, c_A23_0), 2.0);

            // the condition is similar to the rhs but we take abs
            // value of the contributions and add in BH criteria
            Real psi_bl = get_psi_brill_lindquist(loc, a_params);
            Real psi_0 = multigrid_vars_box(iv, c_psi_reg) + psi_bl;
            Real Pi_field = multigrid_vars_box(iv, c_Pi_0);

            condition_box(iv, 0) = 1.5 * abs(m) + //1.5 * A2 * pow(psi_0, -7.0) +
                                   12.0 * M_PI * a_params.G_Newton *
                                       abs(grad_phi_sq) * pow(psi_0, 1.0) +
                                   M_PI * a_params.G_Newton * pow(psi_0, -1.0) * pow(Pi_field, 2.0) +
                                   log(psi_0);
        }
    }
} // end set_regrid_condition

// Add the correction to psi0 after the solver operates
void set_update_psi0(LevelData<FArrayBox> &a_multigrid_vars,
                     LevelData<FArrayBox> &a_dpsi,
                     const Copier &a_exchange_copier)
{

    // first exchange ghost cells for dpsi so they are filled with the correct
    // values
    a_dpsi.exchange(a_dpsi.interval(), a_exchange_copier);

    DataIterator dit = a_multigrid_vars.dataIterator();
    for (dit.begin(); dit.ok(); ++dit)
    {
        FArrayBox &multigrid_vars_box = a_multigrid_vars[dit()];
        FArrayBox &dpsi_box = a_dpsi[dit()];

        Box this_box = multigrid_vars_box.box();
        BoxIterator bit(this_box);
        for (bit.begin(); bit.ok(); ++bit)
        {
            IntVect iv = bit();
            multigrid_vars_box(iv, c_psi_reg) += dpsi_box(iv, 0);
        }
    }
}

// m(K, rho) = 2/3K^2 - 16piG rho
inline Real get_m(const Real &phi_here, const PoissonParameters &a_params,
                  const Real constant_K)
{

    // KC TODO:
    // For now rho is just the gradient term which is kept separate
    // ... may want to add V(phi) and phidot/Pi here later though
    Real V_of_phi = 0.0;
    Real rho = V_of_phi;

    return (2.0 / 3.0) * (constant_K * constant_K) -
           16.0 * M_PI * a_params.G_Newton * rho;
}

// The coefficient of the I operator on dpsi
void set_a_coef(LevelData<FArrayBox> &a_aCoef,
                LevelData<FArrayBox> &a_multigrid_vars,
                const PoissonParameters &a_params, const RealVect &a_dx,
                const Real constant_K)
{

    CH_assert(a_multigrid_vars.nComp() == NUM_MULTIGRID_VARS);

    DataIterator dit = a_aCoef.dataIterator();
    for (dit.begin(); dit.ok(); ++dit)
    {
        FArrayBox &aCoef_box = a_aCoef[dit()];
        FArrayBox &multigrid_vars_box = a_multigrid_vars[dit()];
        Box this_box = aCoef_box.box();

        FArrayBox phi_fab(Interval(c_phi_0, c_phi_0), multigrid_vars_box);

        BoxIterator bit(this_box);
        for (bit.begin(); bit.ok(); ++bit)
        {
            IntVect iv = bit();
            RealVect loc;
            get_loc(loc, iv, a_dx, a_params);
            // m(K, phi) = 2/3 K^2 - 16 pi G rho
            Real m =
                get_m(multigrid_vars_box(iv, c_phi_0), a_params, constant_K);
            Real grad_phi_sq = get_grad_phi_sq(iv, phi_fab, a_dx);

            // Also \bar  A_ij \bar A^ij
            Real A2 = 0.0;
            A2 = pow(multigrid_vars_box(iv, c_A11_0), 2.0) +
                 pow(multigrid_vars_box(iv, c_A22_0), 2.0) +
                 pow(multigrid_vars_box(iv, c_A33_0), 2.0) +
                 2 * pow(multigrid_vars_box(iv, c_A12_0), 2.0) +
                 2 * pow(multigrid_vars_box(iv, c_A13_0), 2.0) +
                 2 * pow(multigrid_vars_box(iv, c_A23_0), 2.0);

            Real psi_bl = get_psi_brill_lindquist(loc, a_params);
            Real psi_0 = multigrid_vars_box(iv, c_psi_reg) + psi_bl;
            Real Pi_field = multigrid_vars_box(iv, c_Pi_0);

            aCoef_box(iv, 0) = -0.625 * m * pow(psi_0, 4.0)
                               //- 0.875 * A2 * pow(psi_0, -8.0)
                               + M_PI * a_params.G_Newton * grad_phi_sq
                               - M_PI * a_params.G_Newton * pow(psi_0, -2.0) * pow(Pi_field, 2.0);
        }
    }
}

// The coefficient of the Laplacian operator, for now set to constant 1
// Note that beta = -1 so this sets the sign
// the rhs source of the Poisson eqn
void set_b_coef(LevelData<FArrayBox> &a_bCoef,
                const PoissonParameters &a_params, const RealVect &a_dx)
{

    CH_assert(a_bCoef.nComp() == 1);
    int comp_number = 0;

    for (DataIterator dit = a_bCoef.dataIterator(); dit.ok(); ++dit)
    {
        FArrayBox &bCoef_box = a_bCoef[dit()];
        bCoef_box.setVal(1.0, comp_number);
    }
}

// used to set output data for all ADM Vars for GRChombo restart
void set_output_data(LevelData<FArrayBox> &a_grchombo_vars,
                     LevelData<FArrayBox> &a_multigrid_vars,
                     const PoissonParameters &a_params, const RealVect &a_dx,
                     const Real &constant_K)
{

    CH_assert(a_grchombo_vars.nComp() == NUM_GRCHOMBO_VARS);
    CH_assert(a_multigrid_vars.nComp() == NUM_MULTIGRID_VARS);

    DataIterator dit = a_grchombo_vars.dataIterator();
    for (dit.begin(); dit.ok(); ++dit)
    {
        FArrayBox &grchombo_vars_box = a_grchombo_vars[dit()];
        FArrayBox &multigrid_vars_box = a_multigrid_vars[dit()];

        // first set everything to zero
        for (int comp = 0; comp < NUM_GRCHOMBO_VARS; comp++)
        {
            grchombo_vars_box.setVal(0.0, comp);
        }

        // now set non zero terms - const across whole box
        // Conformally flat, and lapse = 1
        grchombo_vars_box.setVal(1.0, c_h11);
        grchombo_vars_box.setVal(1.0, c_h22);
        grchombo_vars_box.setVal(1.0, c_h33);
        grchombo_vars_box.setVal(1.0, c_lapse);

        // constant K
        grchombo_vars_box.setVal(constant_K, c_K);

        // now non constant terms by location
        Box this_box = grchombo_vars_box.box();
        BoxIterator bit(this_box);
        for (bit.begin(); bit.ok(); ++bit)
        {
            IntVect iv = bit();
            RealVect loc;
            get_loc(loc, iv, a_dx, a_params);

            // GRChombo conformal factor chi = psi^-4
            Real psi_bl = get_psi_brill_lindquist(loc, a_params);
            Real chi = pow(multigrid_vars_box(iv, c_psi_reg), -4.0);
            grchombo_vars_box(iv, c_chi) = chi;
            Real factor = pow(chi, 1.5);

            // Copy phi, pi, and Aij across - note this is now \tilde Aij not \bar
            // Aij
            grchombo_vars_box(iv, c_phi) = multigrid_vars_box(iv, c_phi_0);
            grchombo_vars_box(iv, c_Pi) = multigrid_vars_box(iv, c_Pi_0);
            grchombo_vars_box(iv, c_A11) =
                multigrid_vars_box(iv, c_A11_0) * factor;
            grchombo_vars_box(iv, c_A12) =
                multigrid_vars_box(iv, c_A12_0) * factor;
            grchombo_vars_box(iv, c_A13) =
                multigrid_vars_box(iv, c_A13_0) * factor;
            grchombo_vars_box(iv, c_A22) =
                multigrid_vars_box(iv, c_A22_0) * factor;
            grchombo_vars_box(iv, c_A23) =
                multigrid_vars_box(iv, c_A23_0) * factor;
            grchombo_vars_box(iv, c_A33) =
                multigrid_vars_box(iv, c_A33_0) * factor;
        }
    }
}

void Hamiltonian_constraint(const Vector<DisjointBoxLayout> &grids, Vector<LevelData<FArrayBox> *> multigrid_vars, const Vector<RealVect> &a_dx,
             const PoissonParameters &a_params, const Real constant_K)
{   
     for (int ilev = 0; ilev < a_params.numLevels; ilev++)
        {
            Real Ham_constraint = 0;
            int Num = 0;
            Real volume = pow(1.5625*32, 3);
            LevelData<FArrayBox> *Fields = multigrid_vars[ilev];
            //LevelData<FArrayBox> *rhs_level_data = rhs[ilev];
            //DataIterator dit = Fields -> dataIterator();
            DataIterator dit = grids[ilev].dataIterator();
            for (dit.begin(); dit.ok(); ++dit)
            {
                FArrayBox& Fields_box = (*Fields)[dit()];
                // FArrayBox& rhs_box = (*rhs_level_data)[dit()];
                FArrayBox psi_fab(Interval(c_psi_reg, c_psi_reg), Fields_box);
                const Box& b = grids[ilev][dit];
                BoxIterator bit(b);
                for (bit.begin(); bit.ok(); ++bit)
                {
                    // Must take the modular coordinate to assign ghost values correctly
                    IntVect iv = bit();
                    Real laplacian_value = get_laplacian_psi(iv, psi_fab, a_dx[ilev]);
                    Num += 1;
                    Ham_constraint += laplacian_value 
                       - (pow(psi_fab(iv), 5) * pow(constant_K, 2) / 12.0 
                       - M_PI * pow(Fields_box(iv, c_Pi_0), 2) * pow(psi_fab(iv), 5));
                }
            }

            pout() << "The Hamiltonian constraint at level " << ilev << " is give by " << Ham_constraint / pow(1.5625*32, 3) << endl;
        }
}

Vector<LevelData<FArrayBox> *> Ham_calc(const Vector<DisjointBoxLayout> &grids, Vector<LevelData<FArrayBox> *> multigrid_vars, const Vector<RealVect> &a_dx,
             const PoissonParameters &a_params, const Real constant_K)
{   
    Vector<LevelData<FArrayBox> *> Ham(multigrid_vars.size(), NULL);

    // Initialize the Hamiltonian constraint girds
    for (int ilev = 0; ilev < a_params.numLevels; ilev++)
    {
        Ham[ilev] = new LevelData<FArrayBox>(grids[ilev], 1, IntVect::Zero);
    }

    // Fill in the Hamiltonian constraint
    for (int ilev = 0; ilev < a_params.numLevels; ilev++)
        {
            LevelData<FArrayBox> *Fields = multigrid_vars[ilev];
            LevelData<FArrayBox> *Ham_fields = Ham[ilev];
            DataIterator dit = grids[ilev].dataIterator();
            for (dit.begin(); dit.ok(); ++dit)
            {
                FArrayBox& Fields_box = (*Fields)[dit()];
                FArrayBox& Ham_box = (*Ham_fields)[dit()];
                FArrayBox psi_fab(Interval(c_psi_reg, c_psi_reg), Fields_box);
                const Box& b = grids[ilev][dit];
                BoxIterator bit(b);
                for (bit.begin(); bit.ok(); ++bit)
                {
                    // Must take the modular coordinate to assign ghost values correctly
                    IntVect iv = bit();
                    Real laplacian_value = get_laplacian_psi(iv, psi_fab, a_dx[ilev]);
                    Ham_box(iv) = laplacian_value 
                       - (pow(psi_fab(iv), 5) * pow(constant_K, 2) / 12.0 
                       - M_PI * pow(Fields_box(iv, c_Pi_0), 2) * pow(psi_fab(iv), 5));
                }
            }
        }
    return Ham;
} // end set_rhs

Vector<LevelData<FArrayBox> *> normalized_Ham_calc(const Vector<DisjointBoxLayout> &grids, Vector<LevelData<FArrayBox> *> multigrid_vars, const Vector<RealVect> &a_dx,
             const PoissonParameters &a_params, const Real constant_K)
{   
    Vector<LevelData<FArrayBox> *> normalized_Ham(multigrid_vars.size(), NULL);

    // Initialize the Hamiltonian constraint girds
    for (int ilev = 0; ilev < a_params.numLevels; ilev++)
    {
        normalized_Ham[ilev] = new LevelData<FArrayBox>(grids[ilev], 1, IntVect::Zero);
    }

    // Fill in the Hamiltonian constraint
    for (int ilev = 0; ilev < a_params.numLevels; ilev++)
        {
            LevelData<FArrayBox> *Fields = multigrid_vars[ilev];
            LevelData<FArrayBox> *Ham_fields = normalized_Ham[ilev];
            DataIterator dit = grids[ilev].dataIterator();
            for (dit.begin(); dit.ok(); ++dit)
            {
                FArrayBox& Fields_box = (*Fields)[dit()];
                FArrayBox& Ham_box = (*Ham_fields)[dit()];
                FArrayBox psi_fab(Interval(c_psi_reg, c_psi_reg), Fields_box);
                const Box& b = grids[ilev][dit];
                BoxIterator bit(b);
                for (bit.begin(); bit.ok(); ++bit)
                {
                    // Must take the modular coordinate to assign ghost values correctly
                    IntVect iv = bit();
                    Real laplacian_value = get_laplacian_psi(iv, psi_fab, a_dx[ilev]);
                    Real Ham_value = laplacian_value 
                       - (pow(psi_fab(iv), 5) * pow(constant_K, 2) / 12.0 
                       - M_PI * pow(Fields_box(iv, c_Pi_0), 2) * pow(psi_fab(iv), 5));
                    Real Ham_ref = sqrt(pow(laplacian_value, 2) 
                      + pow((pow(psi_fab(iv), 5) * pow(constant_K, 2) / 12.0), 2)
                      + pow(M_PI * pow(Fields_box(iv, c_Pi_0), 2) * pow(psi_fab(iv), 5), 2));
                    
                    Ham_box(iv) = Ham_value / Ham_ref; 
                    // pout() << "value " << Ham_value << " and " << Ham_ref << endl;
                    // pout() << "Laplacian " << laplacian_value << endl;
                    // pout() << "Term 1:   " << pow(psi_fab(iv), 5) * pow(constant_K, 2) / 12.0  << endl;
                    // pout() << "Term 2:   " << M_PI * pow(Fields_box(iv, c_Pi_0), 2) * pow(psi_fab(iv), 5) << endl;
     
                }
            }
        }
    return normalized_Ham;
} // end set_rhs
