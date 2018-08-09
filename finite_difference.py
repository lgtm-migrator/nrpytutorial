# finite_difference.py:
#  Sets up finite difference stencils as desired.
#
# Depends on: grid.py.
#             Everything depends on outputC.py.

import grid as gri
import re
from outputC import *

from operator import itemgetter

# Step 1: Initialize free parameters for this module:
import NRPy_param_funcs as par
modulename = __name__
# Centered finite difference accuracy order
par.initialize_param(par.glb_param("INT", modulename, "FD_CENTDERIVS_ORDER",  4))

def FD_outputC(filename,sympyexpr_list):
    # Step 0.a:
    # In case sympyexpr_list is a single sympy expression,
    #     convert it to a list with just one element:
    if type(sympyexpr_list) is not list:
        sympyexpr_list = [sympyexpr_list]

    # Step 0.b:
    # Add proper indentation for C code written by this
    #     function.
    if par.parval_from_str("includebraces") == True:
        indent = "   "
    else:
        indent = ""

    # Step 1:
    # Create a list of free symbols in the sympy expr list
    #     that are registered neither as gridfunctions nor
    #     as C parameters. These *must* be derivatives,
    #     so we call the list "list_of_deriv_vars"
    list_of_deriv_vars_with_duplicates = []
    for expr in sympyexpr_list:
        for var in expr.rhs.free_symbols:
            vartype = gri.variable_type(var)
            if vartype == "other":
                # vartype=="other" should ONLY refer to derivatives, so
                #    if "_dD" or variants do not appear in a variable classified
                #    neither as a gridfunction nor a Cparameter, then error out.
                if "_dD" or "_dKOD" or "_dupD" or "_ddnD" in str(var):
                    pass
                else:
                    print("Error: "+str(var)+" is an unregistered variable.")
                    print("Please check for typos, or register as a gridfunction or Cparameter.")
                    exit(1)
                list_of_deriv_vars_with_duplicates.append(var)
    list_of_deriv_vars = superfast_uniq(list_of_deriv_vars_with_duplicates)

    # Step 2:
    # Process list_of_deriv_vars into a list of base gridfunctions
    #    and a list of derivative operators.
    # Step 2a:
    # First determine the base gridfunction name from
    #     "list_of_deriv_vars"
    deriv__base_gridfunction_name = []
    deriv__operator = []
    for var in list_of_deriv_vars:
        # Step 2a.1: Check that the number of juxtaposed integers
        #            at the end of a variable name matches the
        #            number of U's + D's in the variable name:
        varstr = str(var)
        num_UDs = 0
        for i in range(len(varstr)):
            if varstr[i] == 'D' or varstr[i] == 'U':
                num_UDs += 1
        num_digits = 0
        i = len(varstr) - 1
        while varstr[i].isdigit():
            num_digits += 1
            i-=1
        if num_UDs != num_digits:
            print("Error: "+varstr+" has "+str(num_UDs)+" U's and D's, but ")
            print(str(num_digits)+" integers at the end. These must be equal.")
            print("Please rename your gridfunction.")
            exit(1)

        # Step 2a.2: Based on the variable name, find the rank of
        #            the underlying gridfunction of which we're
        #            trying to take the derivative.
        rank = 0 # rank = "number of juxtaposed U's and D's before the underscore in a derivative expression"
        underscore_position = -1
        for i in range(len(varstr)-1,-1,-1):
            if underscore_position > 0 and (varstr[i] == "U" or varstr[i] == "D"):
                rank += 1
            if varstr[i] == "_":
                underscore_position = i

        # Step 2a.3: Based on the variable name, find the order
        #            of the derivative we're trying to take.
        deriv_order = 0 # deriv_order = "number of D's after the underscore in a derivative expression"
        for i in range(underscore_position+1,len(varstr)):
            if (varstr[i] == "D"):
                deriv_order += 1

        # Step 2a.4: Based on derivative order and rank,
        #            store the base gridfunction name in
        #            deriv__base_gridfunction_name[]
        deriv__base_gridfunction_name.append(varstr[0:underscore_position]+
                                             varstr[len(varstr)-deriv_order-rank:len(varstr)-deriv_order])
        deriv__operator.append(varstr[underscore_position+1:len(varstr)-deriv_order-rank]+
                               varstr[len(varstr)-deriv_order:len(varstr)])

    # Step 2b:
    # Then check each base gridfunction to determine whether
    #     it is indeed registered as a gridfunction.
    #     If not, exit with error.
    for basegf in deriv__base_gridfunction_name:
        is_gf = False
        for gf in gri.glb_gridfcs_list:
            if basegf == str(gf.name):
                is_gf = True
                pass
        if not is_gf:
            print("Error: Attempting to take the derivative of "+basegf+", which is not a registered gridfunction.")
            exit(1)

    # Step 2c:
    # Check each derivative operator to make sure it is
    #     supported. If not, error out.
    for i in range(len(deriv__operator)):
        found_derivID = False
        for derivID in ["dD","dupD","ddnD","dKOD"]:
            if derivID in deriv__operator[i]:
                found_derivID = True
        if not found_derivID:
            print("Error: Valid derivative operator in "+deriv__operator[i]+" not found.")
            exit(1)

    # Step 3:
    # Evaluate the finite difference stencil for each
    #     derivative operator,
    # TODO: being careful not to needlessly recompute.
    # Note: Each finite difference stencil consists
    #     of two parts:
    #     1) The coefficient, and
    #     2) The index corresponding to the coefficient.
    #     The former is stored as a rational number, and
    #     the latter as a simple string, such that e.g.,
    #     in 3D, the empty string corresponds to (i,j,k),
    #     the string "ip1" corresponds to (i+1,j,k),
    #     the string "ip1kp1" corresponds to (i+1,j,k+1),
    #     etc.
    fdcoeffs = [[] for i in range(len(deriv__operator))]
    fdstencl = [[[] for i in range(4)] for j in range(len(deriv__operator))]
    for i in range(len(deriv__operator)):
        fdcoeffs[i], fdstencl[i] = compute_fdcoeffs_fdstencl(deriv__operator[i])

    # Step 4:
    # Create C code to read gridfunctions from memory
    # Step 4a: Compile list of points to read from memory
    #          for each gridfunction i, based on list
    #          provided in fdstencil[i][].
    list_of_points_read_from_memory_with_duplicates = [[] for i in range(len(gri.glb_gridfcs_list))]
    for j in range(len(deriv__base_gridfunction_name)):
        derivgfname = deriv__base_gridfunction_name[j]
        # Next find the corresponding gridfunction index:
        for i in range(len(gri.glb_gridfcs_list)):
            gfname = gri.glb_gridfcs_list[i].name
            # If the gridfunction for the derivative matches, then
            #    add to the list of points read from memory:
            if derivgfname == gfname:
                for k in range(len(fdstencl[j])):
                    list_of_points_read_from_memory_with_duplicates[i].append(str(fdstencl[j][k][0]) + "," + \
                                                                              str(fdstencl[j][k][1]) + "," + \
                                                                              str(fdstencl[j][k][2]) + "," + \
                                                                              str(fdstencl[j][k][3]))

    # Step 4b: "Zeroth derivative" case:
    #     If gridfunction appears in expression not
    #     as derivative (i.e., by itself), it must
    #     be read from memory as well.
    for expr in range(len(sympyexpr_list)):
        for var in sympyexpr_list[expr].rhs.free_symbols:
            vartype = gri.variable_type(var)
            if vartype == "gridfunction":
                for i in range(len(gri.glb_gridfcs_list)):
                    gfname = gri.glb_gridfcs_list[i].name
                    if gfname == str(var):
                        list_of_points_read_from_memory_with_duplicates[i].append("0,0,0,0")

    # Step 4c: Remove duplicates when reading from memory;
    #     do not needlessly read the same variable
    #     from memory twice.
    list_of_points_read_from_memory = [[] for i in range(len(gri.glb_gridfcs_list))]
    for i in range(len(gri.glb_gridfcs_list)):
        list_of_points_read_from_memory[i] = superfast_uniq(list_of_points_read_from_memory_with_duplicates[i])

    # Step 4d: Minimize cache misses:
    #      Sort the list of points read from
    #      main memory by how they are stored
    #      in memory.

    # Step 4d.i: Define a function that maps a gridpoint
    #     index (i,j,k,l) to a unique memory "address",
    #     which will correspond to the correct ordering
    #     of actual memory addresses.
    #
    #     Input: a list of 4 indices, e.g., (i,j,k,l)
    #            corresponding to a gridpoint's *spatial*
    #            index in memory (thus we support up to
    #            4D in space). If spatial dimension is
    #            less than 4D, then just set latter
    #            index/indices to zero. E.g., for 2D
    #            spatial indexing, set (i,j,0,0).
    #     Output: a single number, which when sorted
    #            will yield a unique "address" in memory
    #            such that consecutive addresses are
    #            consecutive in memory.
    def unique_idx(idx4):
        # os and sz are set *just for the purposes of ensuring indices are ordered in memory*
        #    Do not modify the values of os and sz.
        os = 50  # offset
        sz = 100 # assumed size in each direction
        if par.parval_from_str("MemAllocStyle") == "210":
            return str(int(idx4[0])+os + sz*( (int(idx4[1])+os) + sz*( (int(idx4[2])+os) + sz*( int(idx4[3])+os ) ) ))
        elif par.parval_from_str("MemAllocStyle") == "012":
            return str(int(idx4[3])+os + sz*( (int(idx4[2])+os) + sz*( (int(idx4[1])+os) + sz*( int(idx4[0])+os ) ) ))
        else:
            print("Error: MemAllocStyle = "+par.parval_from_str("MemAllocStyle")+" unsupported.")
            exit(1)

    # Step 4d.ii: For each gridfunction and
    #      point read from memory, call unique_idx,
    #      then sort according to memory "address"
    # Input: list_of_points_read_from_memory[gridfunction][point],
    #        gri.glb_gridfcs_list[gridfunction]
    # Output: 1) A list of points to be read from
    #            memory, sorted according to memory
    #            "address":
    #            sorted_list_of_points_read_from_memory[gridfunction][point]
    #        2) A list containing the gridfunction
    #           read at each point, with the number
    #           of elements corresponding exactly
    #           to the total number of points read
    #           from memory for all gridfunctions:
    #           read_from_memory_gf[]
    read_from_memory_gf    = []
    sorted_list_of_points_read_from_memory = [[] for i in range(len(gri.glb_gridfcs_list))]
    for gfidx in range(len(gri.glb_gridfcs_list)):
        # Continue only if reading at least one point of gfidx from memory.
        #     The sorting algorithm at the end of this code block is not
        #     well-defined (will throw an error) if no points of gfidx are
        #     read from memory.
        if len(list_of_points_read_from_memory[gfidx]) > 0:
            read_from_memory_index = []
            for idx in list_of_points_read_from_memory[gfidx]:
                read_from_memory_gf.append(gri.glb_gridfcs_list[gfidx])
                idxsplit = idx.split(',')
                idx4 = [int(idxsplit[0]),int(idxsplit[1]),int(idxsplit[2]),int(idxsplit[3])]
                read_from_memory_index.append(unique_idx(idx4))
                # https://stackoverflow.com/questions/13668393/python-sorting-two-lists
                UNUSEDlist, sorted_list_of_points_read_from_memory[gfidx] = \
                    [list(x) for x in zip(*sorted(zip(read_from_memory_index, list_of_points_read_from_memory[gfidx]),
                                                  key=itemgetter(0)))]

    # Step 4e: Create the full C code string
    #      for reading from memory:

    # if DIM==4:
    #     input: [i,j,k,l]
    #     output: "i0+i,i1+j,i2+k,i3+l"
    # if DIM==3:
    #     input: [i,j,k,l]
    #     output: "i0+i,i1+j,i2+k"
    # etc.
    def ijkl_string(idx4):
        DIM = par.parval_from_str("DIM")
        retstring = ""
        for i in range(DIM):
            if i>0:
                # Add a comma
                retstring += ","
            retstring += "i"+str(i)+"+"+str(idx4[i])
        return retstring.replace("+-", "-").replace("+0", "")

    def vartype():
        if par.parval_from_str("SIMD_enable") == True:
            return "const REAL_SIMD_ARRAY "
        else:
            TYPE = par.parval_from_str("PRECISION")
            return "const " + TYPE + " "

    def varsuffix(idx4):
        if idx4 == [0,0,0,0]:
            return ""
        return "_"+ijkl_string(idx4).replace(",","_").replace("+","p").replace("-","m")

    def read_from_memory_Ccode_onept(gfname,idx):
        idxsplit = idx.split(',')
        idx4 = [int(idxsplit[0]),int(idxsplit[1]),int(idxsplit[2]),int(idxsplit[3])]
        #gfaccess_str = gri.gfaccess("in_gfs",gfname.upper()+"GF",ijkl_string(idx4))
        gfaccess_str = gri.gfaccess("in_gfs",gfname,ijkl_string(idx4))
        if par.parval_from_str("SIMD_enable") == True:
            retstring = indent+vartype() + gfname + varsuffix(idx4) +" = ReadSIMD(&" + gfaccess_str + ");"
        else:
            retstring = indent+vartype() + gfname + varsuffix(idx4) +" = " + gfaccess_str + ";"
        return retstring+"\n"

    read_from_memory_Ccode = ""
    count = 0
    for gfidx in range(len(gri.glb_gridfcs_list)):
        for pt in range(len(sorted_list_of_points_read_from_memory[gfidx])):
            read_from_memory_Ccode += read_from_memory_Ccode_onept(read_from_memory_gf[count].name,
                                                                   sorted_list_of_points_read_from_memory[gfidx][pt])
            count += 1

    # Step 5: Output C code. C code consists of three parts
    #         a) Read gridfunctions from memory at needed pts.
    #         b) Perform arithmetic needed for input expressions
    #            provided in sympyexpr_list[].rhs and associated
    #            finite differences.
    #         c) Write output to gridfunctions specified in
    #            sympyexpr_list[].lhs.

    # Step 5a: Read gridfunctions from memory at needed pts.
    # *** No need to do anything here; already set in
    #     string "read_from_memory_Ccode". ***

    # Step 5b: Perform arithmetic needed for finite differences
    #          associated with input expressions provided in
    #          sympyexpr_list[].rhs.
    #          Note that exprs and lhsvarnames contain
    #          i)  finite difference expressions (constructed
    #              in steps above) and associated variable names,
    #              and
    #          ii) Input expressions sympyexpr_list[], which
    #              in general depend on finite difference
    #              variables.
    exprs    = [sp.sympify(0) for i in range(len(list_of_deriv_vars))]
    lhsvarnames = ["" for i in range(len(list_of_deriv_vars))]
    # Step 5b.i: Add finite difference expressions to
    #            exprs and lhsvarnames lists
    for i in range(len(list_of_deriv_vars)):
        exprs[i]    = sp.sympify(0)
        lhsvarnames[i] = vartype()+str(list_of_deriv_vars[i])
        var = deriv__base_gridfunction_name[i]
        for j in range(len(fdcoeffs[i])):
            varname = str(var)+varsuffix(fdstencl[i][j])
            exprs[i] += fdcoeffs[i][j]*sp.sympify(varname)

        # Multiply each expression by the appropriate power
        #   of 1/dx[i]
        invdx = []
        for d in range(par.parval_from_str("DIM")):
            invdx.append(sp.sympify("invdx"+str(d)))
        # First-order or Kreiss-Oliger derivatives:
        if (len(deriv__operator[i]) == 5 and "dKOD" in deriv__operator[i]) or \
           (len(deriv__operator[i]) == 3 and "dD" in deriv__operator[i]) or \
           (len(deriv__operator[i]) == 5 and ("dupD" in deriv__operator[i] or "ddnD" in deriv__operator[i])):
                dirn = int(deriv__operator[i][len(deriv__operator[i])-1])
                exprs[i] *= invdx[dirn]
        # Second-order derivs:
        elif len(deriv__operator[i]) == 5 and "dDD" in deriv__operator[i]:
            dirn1 = int(deriv__operator[i][len(deriv__operator[i]) - 2])
            dirn2 = int(deriv__operator[i][len(deriv__operator[i]) - 1])
            exprs[i] *= invdx[dirn1]*invdx[dirn2]
        else:
            print("Error: was unable to parse derivative operator: ",deriv__operator[i])
            exit(1)
    # Step 5b.ii: Add input RHS & LHS expressions from
    #             sympyexpr_list[]
    for i in range(len(sympyexpr_list)):
        exprs.append(sympyexpr_list[i].rhs)
        if par.parval_from_str("SIMD_enable") == True:
            lhsvarnames.append("const REAL_SIMD_ARRAY __RHS_exp_"+str(i))
        else:
            lhsvarnames.append(sympyexpr_list[i].lhs)
    # Step 5c: Write output to gridfunctions specified in
    #          sympyexpr_list[].lhs.
    write_to_mem_string = ""
    if par.parval_from_str("SIMD_enable") == True:
        for i in range(len(sympyexpr_list)):
            write_to_mem_string += indent+"WriteSIMD(&"+sympyexpr_list[i].lhs+", __RHS_exp_"+str(i)+");\n"
    Coutput = outputC(exprs,lhsvarnames,"returnstring",CSE_enable=True,prestring=read_from_memory_Ccode,poststring=write_to_mem_string)
    if filename == "stdout":
        print(Coutput)
    elif filename == "returnstring":
        return Coutput+'\n'
    else:
        # Output to the file specified by outCfilename
        with open(filename, par.parval_from_str("outCfileaccess")) as file:
            file.write(final_Ccode_output_str)
        successstr = ""
        if par.parval_from_str("outCfileaccess") == "a":
            successstr = "Appended to "
        elif par.parval_from_str("outCfileaccess") == "w":
            successstr = "Wrote "
        print(successstr + "to file \"" + par.parval_from_str("outCfilename") + "\"")
#    print(gri.glb_gridfcs_list[1].name,list_of_points_read_from_memory[1])


#  Define the to-be-inverted matrix, A.
#  We define A row-by-row, according to the prescription
#  derived in notes/notes.pdf, via the following pattern
#  that applies for arbitrary order.
#
#  As an example, consider a 5-point finite difference
#  stencil (4th-order accurate), where we wish to compute
#  some derivative at the center point.
#
#  Then A is given by:
#
#  -2^0  -1^0  1  1^0   2^0
#  -2^1  -1^1  0  1^1   2^1
#  -2^2  -1^2  0  1^2   2^2
#  -2^3  -1^3  0  1^3   2^3
#  -2^4  -1^4  0  1^4   2^4
#
#  Then right-multiplying A^{-1}
#  by (1 0 0 0 0)^T will yield 0th deriv. stencil
#  by (0 1 0 0 0)^T will yield 1st deriv. stencil
#  by (0 0 1 0 0)^T will yield 2nd deriv. stencil
#  etc.
#
#  Next suppose we want an upwinded, 4th-order accurate
#  stencil. For this case, A is given by:
#
#  -1^0  1  1^0   2^0   3^0
#  -1^1  0  1^1   2^1   3^1
#  -1^2  0  1^2   2^2   3^2
#  -1^3  0  1^3   2^3   3^3
#  -1^4  0  1^4   2^4   3^4
#
#  ... and similarly for the downwinded derivative.
#
#  Finally, let's consider a 3rd-order accurate
#  stencil. This would correspond to an in-place
#  upwind stencil with stencil radius of 2 gridpoints,
#  where other, centered derivatives are 4th-order
#  accurate. For this case, A is given by:
#
#  -1^0  1  1^0   2^0
#  -1^1  0  1^1   2^1
#  -1^2  0  1^2   2^2
#  -1^3  0  1^3   2^3
#  -1^4  0  1^4   2^4
#
#  ... and similarly for the downwinded derivative.
#
#  The general pattern is as follows:
#
#  1) The top row is all 1's,
#  2) If the second row has N elements (N must be odd),
#  .... then the radius of the stencil is rs = (N-1)/2
#  .... and the j'th row e_j = j-rs-1. For example,
#  .... for 4th order, we have rs = 2
#  .... j  | element
#  .... 1  | -2
#  .... 2  | -1
#  .... 3  |  0
#  .... 4  |  1
#  .... 5  |  2
#  3) The L'th row, L>2 will be the same as the second
#  .... row, but with each element e_j -> e_j^(L-1)
#  A1 is used later to validate the inverted
#  matrix.

def compute_fdcoeffs_fdstencl(derivstring,FDORDER=-1):
    # Step 0: Set finite differencing order, stencil size, and up/downwinding
    if FDORDER == -1:
        FDORDER = par.parval_from_str("FD_CENTDERIVS_ORDER")
    STENCILSIZE = FDORDER+1
    UPDOWNWIND = 0
    if "dupD" in derivstring:
        UPDOWNWIND =  1
    elif "ddnD" in derivstring:
        UPDOWNWIND = -1

    # Step 1: Set up matrix based on the stencil size (FDORDER+1).
    #         See documentation above for details on how this
    #         matrix is set up.
    M = sp.zeros(STENCILSIZE,STENCILSIZE)
    for i in range(STENCILSIZE):
        for j in range(STENCILSIZE):
            if i == 0:
                M[(i,j)] = 1 # Setting n^0 = 1 for all n, including n=0, because this matches the pattern
            else:
                dist_from_xeq0_col = j - sp.Rational((STENCILSIZE - 1),2) + UPDOWNWIND
                if dist_from_xeq0_col==0:
                    M[(i,j)] = 0
                else:
                    M[(i, j)] = dist_from_xeq0_col**(i)
    Minv = sp.zeros(STENCILSIZE,STENCILSIZE)
    Minv = M**(-1)

    # Step 2:
    #     Based on the input derivative string,
    #     pick out the relevant row of the matrix
    #     inverse, as outlined in the detailed code
    #     comments prior to this function definition.
    derivtype = "FirstDeriv"
    matrixrow = 1
    if "DDD" in derivstring:
        print("Error: Only derivatives up to second order currently supported.")
        print("       Feel free to contribute to NRPy+ to extend its functionality!")
        exit(1)
    elif "DD" in derivstring:

        if derivstring[len(derivstring)-1] == derivstring[len(derivstring)-2]:
            # Assuming i==j, we call \partial_i \partial_j gf an "unmixed" second derivative,
            #     or more simply, just "SecondDeriv":
            derivtype = "SecondDeriv"
            matrixrow = 2
        else:
            # Assuming i!=j, we call \partial_i \partial_j gf a MIXED second derivative,
            #     which is computed using a composite of first derivative operations.
            derivtype = "MixedSecondDeriv"
    elif "dKOD" in derivstring:
        derivtype = "KreissOligerDeriv"
        matrixrow = STENCILSIZE - 1
    else:
        # Up/downwinded and first derivs are all of "FirstDeriv" type
        pass

    # Step 3:
    #     Set finite difference coefficients
    #     and stencil points corresponding to
    #     each finite difference coefficient.
    fdcoeffs = []
    fdstencl = []
    if derivtype != "MixedSecondDeriv":
        for i in range(STENCILSIZE):
            idx4 = [0, 0, 0, 0]
            # First compute finite difference coefficient.
            fdcoeff = sp.factorial(matrixrow)*Minv[(i,matrixrow)]
            # Do not store fdcoeff or fdstencil if
            # finite difference coefficient is zero.
            if fdcoeff != 0:
                fdcoeffs.append(fdcoeff)
                if derivtype == "KreissOligerDeriv":
                    fdcoeffs[i] *= (-1)**(sp.Rational((STENCILSIZE+1),2))/2**matrixrow

                # Next store finite difference stencil point
                # corresponding to coefficient.
                gridpt_posn = i - int((STENCILSIZE-1)/2) + UPDOWNWIND
                if gridpt_posn != 0:
                    dirn = int(derivstring[len(derivstring)-1])
                    idx4[dirn] = gridpt_posn
                fdstencl.append(idx4)
    else:
        # Mixed second derivative finite difference coeffs
        #     consist of products of first deriv coeffs,
        #     defined in first Minv matrix row.
        for i in range(STENCILSIZE):
            for j in range(STENCILSIZE):
                idx4 = [0, 0, 0, 0]

                # First compute finite difference coefficient.
                fdcoeff = (sp.factorial(matrixrow)*Minv[(i,matrixrow)]) * \
                          (sp.factorial(matrixrow)*Minv[(j,matrixrow)])

                # Do not store fdcoeff or fdstencil if
                # finite difference coefficient is zero.
                if fdcoeff != 0:
                    fdcoeffs.append(fdcoeff)

                    # Next store finite difference stencil point
                    # corresponding to coefficient.
                    gridpt_posn1 = i - int((STENCILSIZE - 1) / 2)
                    gridpt_posn2 = j - int((STENCILSIZE - 1) / 2)
                    dirn1 = int(derivstring[len(derivstring) - 1])
                    dirn2 = int(derivstring[len(derivstring) - 2])
                    idx4[dirn1] = gridpt_posn1
                    idx4[dirn2] = gridpt_posn2
                    fdstencl.append(idx4)
    return fdcoeffs,fdstencl
