/* GRChombo
 * Copyright 2012 The GRChombo collaboration.
 * Please refer to LICENSE in GRChombo's root directory.
 */

#ifdef CH_LANG_CC
/*
 *      _______              __
 *     / ___/ /  ___  __ _  / /  ___
 *    / /__/ _ \/ _ \/  V \/ _ \/ _ \
 *    \___/_//_/\___/_/_/_/_.__/\___/
 *    Please refer to Copyright.txt, in Chombo's root directory.
 */
#endif

#ifndef _WRITEOUTPUT_H_
#define _WRITEOUTPUT_H_

#include "AMRIO.H"
#include "BRMeshRefine.H"
#include "BiCGStabSolver.H"
#include "CH_HDF5.H"
#include "DebugDump.H"
#include "FABView.H"
#include "FArrayBox.H"
#include "GRChomboUserVariables.hpp"
#include "LevelData.H"
#include "LoadBalance.H"
#include "MultigridUserVariables.hpp"
#include "MultilevelLinearOp.H"
#include "ParmParse.H"
#include "PoissonParameters.H"
#include "SetLevelData.H"
#include <iostream>
#include "computeSum.H"


#ifdef CH_Linux
// Should be undefined by default
//#define TRAP_FPE
#undef TRAP_FPE
#endif

#ifdef TRAP_FPE
static void enableFpExceptions();
#endif

using std::cerr;

/// This function outputs the variable data during the solver updates
/// which helps to check convergence and debug
void output_solver_data(const Vector<LevelData<FArrayBox> *> &a_dpsi,
                        const Vector<LevelData<FArrayBox> *> &a_rhs,
                        const Vector<LevelData<FArrayBox> *> &a_multigrid_vars,
                        const Vector<DisjointBoxLayout> &a_grids,
                        const PoissonParameters &a_params, const int iter)
{
#ifdef CH_USE_HDF5

#if CH_SPACEDIM == 2
    string fileName("vcPoissonOut.2d");
#else
    string fileName("vcPoissonOut.3d");
#endif

    char suffix[30];
    sprintf(suffix, "_%d.hdf5", iter);
    fileName += suffix;

    int ndpsiComp = a_dpsi[0]->nComp();
    int nRhsComp = a_rhs[0]->nComp();
    int nMGComp = a_multigrid_vars[0]->nComp();
    int totalComp = ndpsiComp + nRhsComp + nMGComp;
    CH_assert(totalComp == 2 + NUM_MULTIGRID_VARS);

    // set names
    Vector<string> variable_names(totalComp);
    variable_names[0] = "dpsi";
    variable_names[1] = "rhs";
    for (int icomp = 0; icomp < NUM_MULTIGRID_VARS; icomp++)
    {
        variable_names[2 + icomp] =
            MultigridUserVariables::variable_names[icomp];
    }

    // check the domain sizes are the same
    CH_assert(a_dpsi.size() == a_rhs.size());
    CH_assert(a_dpsi.size() == a_multigrid_vars.size());

    Vector<LevelData<FArrayBox> *> tempData(a_dpsi.size(), NULL);
    for (int level = 0; level < a_dpsi.size(); level++)
    {
        tempData[level] = new LevelData<FArrayBox>(a_grids[level], totalComp);
        Interval dpsiComps(0, 0);
        Interval rhsComps(1, 1);
        Interval mgComps(2, NUM_MULTIGRID_VARS + 1);
        a_dpsi[level]->copyTo(a_dpsi[level]->interval(), *tempData[level],
                              dpsiComps);
        a_rhs[level]->copyTo(a_rhs[level]->interval(), *tempData[level],
                             rhsComps);
        a_multigrid_vars[level]->copyTo(a_multigrid_vars[level]->interval(),
                                        *tempData[level], mgComps);
    }

    Real fakeTime = iter * 1.0;
    Real fakeDt = 1.0;
    WriteAMRHierarchyHDF5(fileName, a_grids, tempData, variable_names,
                          a_params.coarsestDomain.domainBox(),
                          a_params.coarsestDx, fakeDt, fakeTime,
                          a_params.refRatio, a_params.numLevels);

    // clean up temporary storage
    for (int level = 0; level < a_dpsi.size(); level++)
    {
        delete tempData[level];
        tempData[level] = NULL;
    }
#endif
}

// Here we correct the Pi to be physical.
void Correct_Pi(const Vector<LevelData<FArrayBox> *> &multigrid_vars, const PoissonParameters &a_params)
{
    for (int ilev = 0; ilev < a_params.numLevels; ilev++)
        {
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
                    Fields_box(iv, c_Pi_0) = Fields_box(iv, c_Pi_0) * pow(Fields_box(iv, c_psi_reg), -3);
                }
            }
        }
}
/*
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

        pout() << a_psi_fab(a_iv, c_psi_reg) << endl;
        pout() << a_psi_fab(iv_offset1, c_psi_reg) << endl;
        pout() << a_psi_fab(iv_offset2, c_psi_reg) << endl;


        // 2nd order stencil for now
        Real d2psi_dxdx = 1.0 / (a_dx[idir] * a_dx[idir]) *
                          (+1.0 * a_psi_fab(iv_offset2, c_psi_reg) -
                           2.0 * a_psi_fab(a_iv, c_psi_reg) + 1.0 * a_psi_fab(iv_offset1, c_psi_reg));
        laplacian_of_psi += d2psi_dxdx;
    }
    return laplacian_of_psi;
} // end get_laplacian_psi

void Hamiltonian_constraint(Vector<LevelData<FArrayBox> *> multigrid_vars, const Vector<RealVect> &a_dx,
             const PoissonParameters &a_params, const Real constant_K)
{

     for (int ilev = 0; ilev < a_params.numLevels; ilev++)
        {
            Real Ham_constraint = 0;
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
                    pout() << "Hello" << endl;
                    IntVect iv = bit();
                    Real laplacian_value = get_laplacian_psi(iv, Fields_box, a_dx[ilev]);
                    pout() << "Laplacian worked" << endl;
                    Ham_constraint += laplacian_value 
                       - (pow(Fields_box(iv, c_psi_reg), 5) * pow(constant_K, 2) / 12.0 
                       - M_PI * pow(Fields_box(iv, c_Pi_0), 2) * pow(Fields_box(iv, c_psi_reg), 5));
                }
            }

            pout() << "The Hamiltonian constraint at level " << ilev << " is give by " << Ham_constraint << endl;
        }
} // end set_rhs
*/
/// This function outputs the final data after the solver is finished
/// in a format that GRChombo can read in for an AMRTimeDependent run
void output_final_data(const Vector<LevelData<FArrayBox> *> &a_multigrid_vars,
                       const Vector<DisjointBoxLayout> &a_grids,
                       const Vector<RealVect> &a_vectDx,
                       const Vector<ProblemDomain> &a_vectDomains,
                       const PoissonParameters &a_params,
                       const Real &constant_K)
{
#ifdef CH_USE_HDF5

#if CH_SPACEDIM == 2
    string filename("InitialConditionsFinal.2d.hdf5");
#else
    string filename("InitialConditionsFinal.3d.hdf5");
#endif

    Correct_Pi(a_multigrid_vars, a_params);
    // Hamiltonian_constraint(a_grids, a_multigrid_vars, a_vectDx, a_params, constant_K);
    Vector<LevelData<FArrayBox> *> Ham = normalized_Ham_calc(a_grids, a_multigrid_vars, a_vectDx, a_params, constant_K);

    int comp_val = 0;
    Real integral = computeSum(Ham, a_params.refRatio, a_params.coarsestDx, Interval(comp_val, comp_val));
 
    pout() << "---------------------------" << endl;
    pout() << "Hamiltonian constraint is: " << integral << endl;

    // Hamiltonian_constraint(a_grids, a_multigrid_vars, a_vectDx, a_params, constant_K);
    // set up full data and write to hdf5 file ready for GRChombo
    HDF5Handle handle(filename, HDF5Handle::CREATE);
    Vector<LevelData<FArrayBox> *> grchombo_vars(a_multigrid_vars.size(), NULL);
    HDF5HeaderData header;

    header.m_int["max_level"] = a_params.maxLevel;
    header.m_int["num_levels"] = a_params.maxLevel + 1;
    header.m_int["iteration"] = 0;
    header.m_real["time"] = 0.0;

    for (int level = 0; level < a_multigrid_vars.size(); ++level)
    {
        char headername1[100];
        sprintf(headername1, "regrid_interval_%d", level);
        header.m_int[headername1] = 1;

        char headername2[100];
        sprintf(headername2, "steps_since_regrid_%d", level);
        header.m_int[headername2] = 0;
    }

    header.m_int["num_components"] = NUM_GRCHOMBO_VARS;

    char comp_str[30];
    for (int comp = 0; comp < NUM_GRCHOMBO_VARS; ++comp)
    {
        sprintf(comp_str, "component_%d", comp);
        header.m_string[comp_str] = GRChomboUserVariables::variable_names[comp];
    }

    // write header data to file
    header.writeToFile(handle);

    // loop over levels
    for (int level = 0; level < a_multigrid_vars.size(); level++)
    {
        HDF5HeaderData level_header;
        // want output data to have 3 ghost cells to match GRChombo
        grchombo_vars[level] = new LevelData<FArrayBox>(
            a_grids[level], NUM_GRCHOMBO_VARS, 3 * IntVect::Unit);

        // Add in the values of the data in the correct form
        set_output_data(*grchombo_vars[level], *a_multigrid_vars[level],
                        a_params, a_vectDx[level], constant_K);

        // Now continue the write out
        char level_str[20];
        sprintf(level_str, "%d", level);
        const std::string label = std::string("level_") + level_str;
        handle.setGroup(label);

        // assign arbitrary values to the params
        // most can be overwritten in GRChombo setup
        level_header.m_int["ref_ratio"] = a_params.refRatio[level];
        level_header.m_int["tag_buffer_size"] = 3;
        level_header.m_real["dx"] = a_vectDx[level][0];
        level_header.m_real["dt"] = 0.25 * a_vectDx[level][0];
        level_header.m_real["time"] = 0.0;
        level_header.m_box["prob_domain"] = a_vectDomains[level].domainBox();

        // Setup the periodicity info - for GRChombo we always treat it as
        // periodic
        for (int dir = 0; dir < SpaceDim; ++dir)
        {
            char dir_str[20];
            sprintf(dir_str, "%d", dir);
            const std::string periodic_label =
                std::string("is_periodic_") + dir_str;
            level_header.m_int[periodic_label] = true;
        }

        level_header.writeToFile(handle);
        write(handle, a_grids[level]);
        write(handle, *grchombo_vars[level], "data");
    }

    // shut the file
    handle.close();

    // clean up temporary storage
    for (int level = 0; level < a_multigrid_vars.size(); level++)
    {
        delete grchombo_vars[level];
        grchombo_vars[level] = NULL;
    }

#endif
}

#endif