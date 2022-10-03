# This module provides functions for setting up the right-hand sides
#     of the evolution equations a massless Scalar Field, i.e. the
#     Klein-Gordon equations, as documented in
#     Tutorial-ScalarField_RHSs.ipynb

# Authors: Leonardo R. Werneck
#          wernecklr **at** gmail **dot* com
#          Zachariah B. Etienne

# Step 0: Import core Python/NRPy+ modules
from sympy import sympify                        # SymPy: The Python computer algebra package upon which NRPy+ depends
from NRPy_param_funcs import set_parval_from_str # NRPy+: Parameter interface
from indexedexp import declarerank1              # NRPy+: Symbolic indexed expression (e.g., tensors, vectors, etc.) support
from reference_metric import reference_metric    # NRPy+: Reference metric support
import BSSN.BSSN_quantities as Bq                # NRPy+: BSSN quantities
import BSSN.ADM_in_terms_of_BSSN as AitoB        # NRPy+: ADM quantities in terms of BSSN quantities
import ScalarField.ScalarField_quantities as SFq # NRPyCritCol: Scalar field gridfunctions and derivatives

def ScalarField_RHSs():

    # Step 1: Basic setup
    # Step 1.a: Set spatial dimension (must be 3 for BSSN, as BSSN is
    #           a 3+1-dimensional decomposition of the general
    #           relativistic field equations)
    DIM = 3

    # Step 1.b: Declare all needed ADM quantities in
    #           terms of BSSN quantities
    Bq.BSSN_basic_tensors()
    trK   = Bq.trK
    alpha = Bq.alpha
    betaU = Bq.betaU
    
    AitoB.ADM_in_terms_of_BSSN()
    gammaDD  = AitoB.gammaDD
    gammaUU  = AitoB.gammaUU
    GammaUDD = AitoB.GammaUDD
    
    # Step 1.c: Declare all needed scalar field quantities
    SFq.ScalarField_quantities()
    sf       = SFq.sf
    sf_dD    = SFq.sf_dD
    sf_dupD  = SFq.sf_dupD
    sf_dDD   = SFq.sf_dDD
    sfM      = SFq.sfM
    sfM_dD   = SFq.sfM_dD
    sfM_dupD = SFq.sfM_dupD

    # Step 2: Computing the RHSs
    global sf_rhs, sfM_rhs

    # Step 2.a: Add Term 1 to sf_rhs: -alpha*Pi
    sf_rhs = - alpha * sfM

    # Step 2.b: Add Term 2 to sf_rhs: beta^{i}\partial_{i}\varphi
    for i in range(DIM):
        sf_rhs += betaU[i] * sf_dupD[i]

    # Step 3a: Add Term 1 to sfM_rhs: alpha * K * Pi
    sfM_rhs = alpha * trK * sfM

    # Step 3b: Add Term 2 to sfM_rhs: beta^{i}\partial_{i}Pi
    for i in range(DIM):
        sfM_rhs += betaU[i] * sfM_dupD[i]

    # Step 3c: Adding Term 3 to sfM_rhs
    # Step 3c.i: Term 3a: -gammabar^{ij}(alpha_{,i}\varphi_{,j} + alpha\varphi_{,ij})
    alpha_dD    = declarerank1("alpha_dD")
    sfMrhsTerm3 = sympify(0)
    for i in range(DIM):
        for j in range(DIM):
            sfMrhsTerm3 += - gammaUU[i][j] * ( alpha_dD[i] * sf_dD[j] + alpha * sf_dDD[i][j] )
    
    # Step 3c.ii: Term 3b: gamma^{ij}\alpha\Gamma^{k}_{ij}\varphi_{,k}
    for i in range(DIM):
        for j in range(DIM):
            for k in range(DIM):
                sfMrhsTerm3 += gammaUU[i][j] * alpha * GammaUDD[k][i][j] * sf_dD[k]
    
    # Step 3c.iii: Add Term 3 to sfM_rhs
    sfM_rhs += sfMrhsTerm3

