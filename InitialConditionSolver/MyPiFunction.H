#include <cmath>
#ifndef MYPIFUNCTION_HPP
#define MYPIFUNCTION_HPP

Real my_pi_function(RealVect loc, Real amplitude, Real wavelength, RealVect L)
{
    //Here we have some function to give me the Pi_field value at position loc.
    Real r2 = D_TERM(loc[0] * loc[0], +loc[1] * loc[1], +loc[2] * loc[2]);
    return 0.1 + 0.01 * (sin(2 * M_PI * loc[0] * wavelength / L[0]) + sin(2 * M_PI * loc[1] * wavelength / L[1]) + sin(2 * M_PI * loc[2] * wavelength / L[2]));
    // return amplitude * exp(-r2 / wavelength);
}

#endif /* MYPIFUNCTION_HPP */
