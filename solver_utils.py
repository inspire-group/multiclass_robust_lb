import numpy as np
from cvxopt import solvers, matrix,  sparse, spmatrix, spdiag, mul, log
from cvxopt.glpk import ilp

# ============================ INCIDENCE HELPER =======================================
def construct_incidence(num_edges_updated, num_hyper, hyper, num_four_hyper, four_hyper, edge_pairs, del_edges, del_edges_four):
    '''Construct incidence matrix from edges and hyperedges'''
    incidence_col_coord = np.zeros(2 * num_edges_updated + 3 * num_hyper + 4 * num_four_hyper).astype(int)
    incidence_row_coord = np.zeros(2 * num_edges_updated + 3 * num_hyper + 4 * num_four_hyper).astype(int)
    cnt = 0
    ir_cnt = 0
    for e in edge_pairs:
        if e not in del_edges:
            incidence_col_coord[cnt] = e[0]
            incidence_col_coord[cnt+1] = e[1]
            incidence_row_coord[cnt] = ir_cnt
            incidence_row_coord[cnt+1] = ir_cnt
            cnt += 2
            ir_cnt += 1
    for (va, vb, vc) in hyper:
        if (va, vb, vc) not in del_edges_four:
            incidence_col_coord[cnt] = va
            incidence_col_coord[cnt+1] = vb
            incidence_col_coord[cnt+2] = vc
            
            incidence_row_coord[cnt] = ir_cnt
            incidence_row_coord[cnt+1] = ir_cnt
            incidence_row_coord[cnt+2] = ir_cnt
            cnt += 3
            ir_cnt += 1
    for (va, vb, vc, vd) in four_hyper:
        incidence_col_coord[cnt] = va
        incidence_col_coord[cnt+1] = vb
        incidence_col_coord[cnt+2] = vc
        incidence_col_coord[cnt+3] = vd
            
        incidence_row_coord[cnt] = ir_cnt
        incidence_row_coord[cnt+1] = ir_cnt
        incidence_row_coord[cnt+2] = ir_cnt
        incidence_row_coord[cnt+3] = ir_cnt
        cnt += 4
        ir_cnt += 1
    return incidence_row_coord, incidence_col_coord

def construct_constraints(v, n_rows, incidence_row_coord, incidence_col_coord, remove_redundant, with_edges):
    '''Append q < 1 constraints to incidence matrix'''
    if remove_redundant:
        eye_row_coords = np.zeros(v).astype(int)
        eye_col_coords = np.zeros(v).astype(int)
        coords_counter = 0
        for i in range(v):
            if i not in with_edges:
                eye_row_coords[coords_counter] = n_rows + i
                eye_col_coords[coords_counter] = i
                coords_counter += 1
        eye_row_coords = eye_row_coords[:coords_counter]
        eye_col_coords = eye_col_coords[:coords_counter]
    else:
        eye_row_coords = np.arange(n_rows, n_rows + v)
        eye_col_coords = np.arange(v)

    incidence_row_coord = np.concatenate((incidence_row_coord, eye_row_coords))
    incidence_col_coord = np.concatenate((incidence_col_coord, eye_col_coords))
    return incidence_row_coord, incidence_col_coord


def minll(G,h,p,v):
    m,v_in=G.size
    def F(x=None,z=None):
        if x is None:
            return 0, matrix(1.0,(v,1))
        if min(x)<=0.0:
            return None
        f = -sum(mul(p,log(x)))
        Df = mul(p.T,-(x**-1).T)
        if z is None:
            return f,Df
        # Fix the Hessian
        H = spdiag(z[0]*mul(p,x**-2))
        return f,Df,H
    return solvers.cp(F,G=G,h=h)

def run_solver(incidence_row_coord, incidence_col_coord, loss, v, n_rows, mosek, run_nonconvex, run_integer):
    p = (1.0/v)*np.ones((v,1))
    G_in_sparse=spmatrix(1.0,incidence_row_coord,incidence_col_coord, size=(n_rows + v, v))
    print('Constraint matrix size = {} x {}'.format(n_rows+v, v))
    solvers.options['maxiters']=10000

    if loss == '0-1':
        c=matrix(-1 * p)
        identity_sparse = spmatrix(-1.0, range(v), range(v))
        A = sparse([identity_sparse, G_in_sparse])
        b = matrix(np.concatenate((np.zeros(v),np.ones(n_rows+v))))
        if run_integer:
            return ilp(c, A, b, B=set(range(v)))
        if mosek:
            return solvers.lp(c, A, b, solver='mosek')
        return solvers.lp(c, A, b)
    else:
        # otherwise it is cross entropy
        if run_nonconvex:
            h_in=np.ones((n_rows+v,1))
            # generic solver
            return minll(G_in_sparse,matrix(h_in),matrix(p),v)
        else:
            # geometric program

            # iterate through incidence_row_cord to match single constraint per row
            # each element of K after the first is the number of vertices within each edge/hyperedge
            K = [1]
            prev_row_coord = None
            for i in range(len(incidence_row_coord)):
                if incidence_row_coord[i] != prev_row_coord:
                    K.append(1)
                    prev_row_coord = incidence_row_coord[i]
                else:
                    K[-1] += 1
                incidence_row_coord[i] = i
            
            F0 = matrix(-1 * p.T)
            # edge constraints
            Fi = spmatrix(1.0,incidence_row_coord,incidence_col_coord, size=(np.sum(K[1:]), v))
            F = sparse([F0, Fi])
            g = matrix(np.zeros(np.sum(K)))

            # < 1 constraints
            G = spmatrix(1.0, range(v), range(v))
            h = matrix(np.zeros(v))

            sol = solvers.gp(K, F, g, G=G, h=h)
            return sol
        

# ===========================DEBUGGING TESTS===================================
def test_incidence(edge_list,hyper_list, v):
    nrows = len(edge_list)+len(hyper_list)
    incidence_row_coord, incidence_col_coord = construct_incidence(len(edge_list), len(hyper_list), hyper_list, edge_list, set())
    incidence_row_coord, incidence_col_coord = construct_constraints(v, nrows, 
                                                incidence_row_coord, incidence_col_coord, False, set())
    p = (1.0/v)*np.ones((v,1))
    c=matrix(-1 * p)
    G_in_sparse=spmatrix(1.0,incidence_row_coord,incidence_col_coord, size=(nrows + v, v))
    identity_sparse = spmatrix(-1.0, range(v), range(v))
    A = sparse([identity_sparse, G_in_sparse])

    b = matrix(np.concatenate((np.zeros(v),np.ones(nrows+v))))
    solvers.options['maxiters']=1000
    
    # run solver
    output= solvers.lp(c, A, b)
    print("optimal 0-1 loss:", 1-sum(np.array(output['x'])*p)[0])
    print(output['x'])
