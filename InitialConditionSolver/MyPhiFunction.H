/* GRChombo
 * Copyright 2012 The GRChombo collaboration.
 * Please refer to LICENSE in GRChombo's root directory.
 */

#include <cmath>
#ifndef MYPHIFUNCTION_HPP
#define MYPHIFUNCTION_HPP

Real my_phi_function(RealVect loc, Real amplitude, Real wavelength, RealVect L)
{
    // gaussian - use with Dirichlet BCs
    //Real r2 = D_TERM(loc[0] * loc[0], +loc[1] * loc[1], +loc[2] * loc[2]);
    //return amplitude * exp(-r2 / wavelength);

    // sine waves in 3 directions - use with periodic BCs
    // return amplitude * (sin(2*M_PI * loc[0] * wavelength / L[0]) + sin(2*M_PI * loc[1] * wavelength / L[1]) + sin(2*M_PI * loc[2] * wavelength / L[2]));
    return 0;
}

#endif /* MYPHIFUNCTION_HPP */
