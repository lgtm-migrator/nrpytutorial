# Step 0: Add NRPy's directory to the path
# https://stackoverflow.com/questions/16780014/import-file-from-parent-directory
import os,sys
nrpy_dir_path = os.path.join("..")
if nrpy_dir_path not in sys.path:
    sys.path.append(nrpy_dir_path)

# Step 0.a: Import the NRPy+ core modules and set the reference metric to Cartesian
from outputC import nrpyAbs
import NRPy_param_funcs as par   # NRPy+: Parameter interface
import indexedexp as ixp         # NRPy+: Symbolic indexed expression (e.g., tensors, vectors, etc.) support
import sympy as sp               # SymPy: The Python computer algebra package upon which NRPy+ depends
import GiRaFFEfood_NRPy.GiRaFFEfood_NRPy_Common_Functions as gfcf # Some useful functions for GiRaFFE initial data.
import reference_metric as rfm   # NRPy+: Reference metric support
par.set_parval_from_str("reference_metric::CoordSystem","Cartesian")
rfm.reference_metric()
# Step 1a: Set commonly used parameters.
thismodule = __name__

nrpyDilog = sp.Function('nrpyDilog')
from outputC import custom_functions_for_SymPy_ccode
custom_functions_for_SymPy_ccode["nrpyDilog"] = "gsl_sf_dilog"

C_SM = par.Cparameters("REAL",thismodule,["C_SM"], 1.0)

def f_of_r(r,M):
    x = sp.sympify(2)*M/r
    L = nrpyDilog(x) + sp.Rational(1,2)*sp.log(x)*sp.log(sp.sympify(1)-x)
    f = r*r*(sp.sympify(2)*r-sp.sympify(3)*M)*sp.Rational(1,8)/(M**3)*L\
       +(M*M+sp.sympify(3)*M*r-sp.sympify(6)*r*r)*sp.Rational(1,12)/(M*M)*sp.log(r*sp.Rational(1,2)/M)\
       +sp.Rational(11,72) + M*sp.Rational(1,3)/r + r*sp.Rational(1,2)/M - r*r/(M*M)
    return f

def fp_of_r(r,M):
    x   = sp.sympify(2)*M/r
    L   = nrpyDilog(x) + sp.Rational(1,2)*sp.log(x)*sp.log(sp.sympify(1)-x)
    Lp  = -sp.Rational(1,2) * (sp.log(sp.sympify(1)-x)/x + sp.log(x)/(sp.sympify(1)-x))
    fp  = (sp.sympify(6)*r*r-sp.sympify(3)*M)*sp.Rational(1,8)/(M**3) + (sp.sympify(2)*r-sp.sympify(3)*M)*sp.Rational(1,4)/(M*M)*Lp\
         +(sp.sympify(3)*M-12*r)*sp.Rational(1,12)/(M*M)*sp.log(r*sp.Rational(1,2)/M) + (M*M+sp.sympify(3)*M*r-sp.sympify(6)*r*r)*sp.Rational(1,3)/r\
         -M*sp.Rational(1,3)/(r*r) + sp.Rational(1,2)/M - 2*r/(M*M)
    return fp

def Ar_SM(r,theta,phi, **params):
    M = params["M"]
    a = params["a"]
    # A_r = -aC/8 * cos \theta ( 1 + 4M/r )
    return -a*C_SM*sp.Rational(1,8)*nrpyAbs(sp.cos(theta))*(sp.sympify(1)+sp.sympify(4)*M/r)

def Ath_SM(r,theta,phi, **params):
    # A_\theta = 0
    return sp.sympify(0)

def Aph_SM(r,theta,phi, **params):
    M = params["M"]
    a = params["a"]
    # A_\phi = M^2 C [1-\cos \theta + a^2 f(r) cos \theta sin^2 \theta]
    return M*M*C_SM*(sp.sympify(2)-nrpyAbs(sp.cos(theta))+a*a*f_of_r(r,M)*sp.cos(theta)*sp.sin(theta)**2)

def ValenciavU_func_SM(**params):
    M = params["M"]
    a = params["a"]
    alpha = params["alpha"]
    betaU = params["betaU"] # Note that this must use a spherical basis!
    gammaDD = params["gammaDD"] # Note that this must use a Cartesian basis!
    sqrtgammaDET = params["sqrtgammaDET"]
    KerrSchild_radial_shift = params["KerrSchild_radial_shift"]
    r     = rfm.xxSph[0] + KerrSchild_radial_shift # We are setting the data up in Shifted Kerr-Schild coordinates
    theta = rfm.xxSph[1]
    phi   = rfm.xxSph[2]

    global BsphU
    BsphU = ixp.zerorank1()
    BsphU[0] = C_SM*alpha*M*M/(r*r) + \
               C_SM*alpha*a*a*M*M*sp.Rational(1,2)/(r**4)*(-sp.sympify(2)*sp.cos(theta) + (r/M)**2*(sp.sympify(1)+sp.sympify(3)*sp.cos(sp.sympify(2)*theta))*f_of_r(r,M))
    BsphU[1] = -C_SM*alpha*a*a/(r*r) * sp.sin(theta)*sp.cos(theta)*fp_of_r(r,M)
    BsphU[2] = -C_SM*alpha*a*M*sp.Rational(1,8)/(r*r)*(sp.sympify(1)+sp.sympify(4)*M/r)

    EsphD = ixp.zerorank1()
    EsphD[0] = -C_SM*a**3/(sp.sympify(8)*alpha*M**3)*fp_of_r(r,M)*sp.cos(theta)*sp.sin(theta)**2
    EsphD[1] = -C_SM*a*sp.Rational(1,8)/alpha*(sp.sin(theta) + a*a*f_of_r(r,M)*sp.sin(theta)*(sp.sympify(2)*sp.cos(theta)**2-sp.sin(theta)**2)) - \
               betaU[0]*sqrtgammaDET*a*C_SM*sp.Rational(1,8)/(r*r)*(sp.sympify(1)+sp.sympify(4)*M/r)
    EsphD[2] = betaU[0]/(alpha*M)*C_SM*a*a*fp_of_r(r,M)*sp.cos(theta)*sp.sin(theta)**2

    ED = gfcf.change_basis_spherical_to_Cartesian_D(EsphD)
    BU = gfcf.change_basis_spherical_to_Cartesian_U(BsphU)

    return gfcf.compute_ValenciavU_from_ED_and_BU(ED, BU, gammaDD)