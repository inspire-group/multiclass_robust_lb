import seaborn as sns
import matplotlib.pyplot as plt
from math import sqrt
import numpy as np
import os


# ====================== HYPEREDGE COMPUTATION HELPERS =======================================
def acutetri(a, b, c):
    '''Compute whether triangle is acute or not and length of largest edge if not acute'''
    if a * a + b * b - c * c < 0:
        return(False, c)
    elif a * a - b * b + c * c < 0:
        return (False, b)
    elif -a * a + b * b + c * c < 0:
        return (False, a)
    return (True, None)
    

def circumradius(a, b, c):
    '''Compute circumradius'''
    numerator = (a * a)*(b * b)*(c * c)
    denomimator = 2*(a * a)*(b * b) + 2*(a * a)*(c * c) + 2*(b * b)*(c * c) - a**4 - b**4 - c**4
    return sqrt(numerator/denomimator)

def bin_search_edge(tofind, edges, startidx, endidx):
    '''Binary search to determine whether tofind is in edges'''
    if endidx < startidx:
        return False
    
    else:
        mid = startidx + (endidx - startidx) // 2
        if edges[mid] == tofind:
            return True
 
        elif edges[mid][1] > tofind[1]:
            return bin_search_edge(tofind, edges, startidx, mid - 1)
 
        else:
            return bin_search_edge(tofind, edges, mid + 1, endidx)

def get_class_idx(va, subsample_size):
    cumsum = np.cumsum(subsample_size)[:-1]
    cumsum = [0] + list(cumsum)
    for i in range(len(cumsum)-1, -1, -1):
        if va >= cumsum[i]:
            return i, va - cumsum[i]


def get_dist(X_cs, va, vb, norm, subsample_size):
    '''Compute distance between vertices va and vb'''
    return np.linalg.norm(X_cs[va][0].reshape(-1)/255 - X_cs[vb][0].reshape(-1)/255, ord=float(norm[1:]))

def verify_hyper(ab, bc, ac, compute_dists, dists, eps, norm, subsample_size):
    '''Check whether triangle ab, bc, ac is also a hyperedge
    If compute_dists is True, dists stores X_cs.  Otherwise dists
    is a dictionary with precomputed distances for each edge'''
    if norm == 'l2':
        if compute_dists:
            dist_ab = get_dist(dists, ab[0], ab[1], norm, subsample_size)
            dist_bc = get_dist(dists, bc[0], bc[1], norm, subsample_size)
            dist_ac = get_dist(dists, ac[0], ac[1], norm, subsample_size)
        else:
            dist_ab = dists[ab]
            dist_bc = dists[bc]
            dist_ac = dists[ac]
        (acute, longest) = acutetri(dist_ab, dist_bc, dist_ac)
        if acute == True:
            circumrad = circumradius(dist_ab, dist_bc, dist_ac)
        else:
            circumrad = longest/2
        return circumrad <= eps
    return True # otherwise it is linf, and all triangles in linf are also hyperedges


def find_hyper(edge_pairs, compute_dists, compute_hyper, num_classes, subsample_size, eps, norm,
                 dists=None, plot_edges=False, Xs=None):
    '''Find hyperedges from list of edges'''
    edge_pairs.sort()
    v = sum(subsample_size)

    start_idx = {} # note edge_pairs is sorted, start_idx keeps track of 
                   # where each starting vertex first appears within edge_pairs
                   # ie) start_idx[a] keeps track of the first index where an edge
                   # of form (a, b) occurs

    if compute_dists:
        end_idx = {} # similar to start_idx, but keeps track of last index
    
    hyper = [] # store hyperedges
    four_hyper = []
    del_edges = set() # store redundant edge constraints
    del_edges_four = set()
    num_triangles = 0
    num_hyper = 0
    num_four_hyper = 0
    num_tetra = 0
    tri_degree_vwise = None
    hyper_degree_vwise = None
    if plot_edges:
        tri_degree_vwise = np.zeros(v)
        hyper_degree_vwise = np.zeros(v)

    if compute_hyper > 2 and num_classes >= 3:
        # initialize start_idx (and end_idx) by iterating once through edge_pairs
        curr = -1 
        for i, edge in enumerate(edge_pairs):
            if compute_dists and edge[0] != curr and i != 0:
                end_idx[curr] = i-1
            if edge[0] != curr:
                curr = edge[0]
                start_idx[curr] = i
        if compute_dists:
            end_idx[curr] = len(edge_pairs) - 1

        # start hyperedge search
        # walk through each edge (a, b)
        for i, ab in enumerate(edge_pairs):
            if ab[1] not in start_idx: # edges are sorted, so this would only occur if
                continue               # if all edges to b have start at a vertex < b
                                       # in this case, we would have already found
                                       # whether it is part of a triangle or not
            
            # walk through edges from b
            for j in range(start_idx[ab[1]], len(edge_pairs)): 
                bc = edge_pairs[j] 

                # check b is the same as previous (i.e ab & bc have the same b )
                if bc[0] != ab[1]: break;   # if not quit finding matches for ab

                ac = (ab[0], bc[1]) 
                # does ac exist? just look in dist dictionary (if we have saved it)
                if not compute_dists and ac not in dists: continue

                # if we didn't save a dictionary of distances corresponding to each edge,
                # since our edge pairs are sorted, we know that ac must lie between the
                # index of ab (i) and the end index of edges beginning with a, 
                # so we can do a binary search to find if c exists
                if compute_dists and not bin_search_edge(ac, edge_pairs, i+1, end_idx[ac[0]]): continue
                
                # ac exists, so must be a triangle
                num_triangles += 1
                if plot_edges:
                    tri_degree_vwise[ab[0]] += 1
                    tri_degree_vwise[ab[1]] += 1
                    tri_degree_vwise[bc[1]] += 1
                
                # check if it is a hyperedge
                if verify_hyper(ab, bc, ac, compute_dists, dists, eps, norm, subsample_size):
                    num_hyper += 1
                    hyper.append((ab[0], ab[1], bc[1])) 
                    del_edges.add(ab)
                    del_edges.add(bc)
                    del_edges.add(ac)
                    if plot_edges:
                        hyper_degree_vwise[ab[0]] += 1
                        hyper_degree_vwise[ab[1]] += 1
                        hyper_degree_vwise[bc[1]] += 1
        four_hyper, del_edges_four, num_four_hyper, num_tetra = four_way_hyper(hyper,  compute_dists, compute_hyper, num_classes, subsample_size, eps, norm, dists, Xs=Xs)
    return hyper, four_hyper, del_edges, del_edges_four, num_triangles, num_hyper, num_tetra, num_four_hyper, tri_degree_vwise, hyper_degree_vwise


def find_hyper_vwise(edge_pairs, compute_dists, compute_hyper, num_classes, subsample_size, eps, norm,
                 dists=None, plot_edges=False):
    v = sum(subsample_size)
    adjacency_dict = dict() # adjacency_dict[a] stores a set of vertices with an edge to a
    checked = set()
    del_edges = set() # store redundant edge constraints
    num_triangles = 0
    num_hyper = 0
    num_tetra = 0
    num_four_hyper = 0
    triangle_degree = None
    hyper=[]
    hyper_degree = None
    if plot_edges:
        triangle_degree = np.zeros(v)
        hyper_degree = np.zeros(v)

    if compute_hyper > 2 and num_classes >= 3:
        # iterate through edge_pairs to construct adjacency_dict
        for edge in edge_pairs:
            v1, v2 = edge
            if v1 in adjacency_dict.keys():
                adjacency_dict[v1].add(v2)
            else:
                adjacency_dict[v1] = {v2}
            if v2 in adjacency_dict.keys():
                adjacency_dict[v2].add(v1)
            else:
                adjacency_dict[v2] = {v1}

        for va in adjacency_dict.keys():
            if len(adjacency_dict[va]) >= 2: # can only be part of a triangle if there are at least 2 edges
                adjacent_vertices = list(adjacency_dict[va])
                num_adjacencies = len(adjacent_vertices)

                # check every pair of adjacent vertices (we know (va, vb) exists and (va, vc) exists)
                for i in range(num_adjacencies-1):
                    vb = adjacent_vertices[i]
                    for j in range(i+1, num_adjacencies):
                        vc = adjacent_vertices[j]

                        # check that (vb, vc) also exists, if it does, then we have a triangle
                        if vc in adjacency_dict[vb]:
                            sorted_vertices = sorted([va, vb, vc])
                            cur_abc = str(sorted_vertices[0])+"_"+str(sorted_vertices[1])+"_"+str(sorted_vertices[2])
                            
                            if cur_abc not in checked:
                                checked.add(cur_abc)
                                num_triangles += 1
                                if plot_edges:
                                    triangle_degree[va] += 1
                                    triangle_degree[vb] += 1
                                    triangle_degree[vc] += 1
                                
                                ab = (sorted_vertices[0], sorted_vertices[1])
                                bc = (sorted_vertices[1], sorted_vertices[2])
                                ac = (sorted_vertices[0], sorted_vertices[2])
                                
                                # check if it is a hyperedge
                                if verify_hyper(ab, bc, ac, compute_dists, dists, eps, norm, subsample_size):
                                    num_hyper += 1
                                    hyper.append((ab[0], ab[1], bc[1])) 
                                    del_edges.add(ab)
                                    del_edges.add(bc)
                                    del_edges.add(ac)
                                    if plot_edges:
                                        hyper_degree[va] += 1
                                        hyper_degree[vb] += 1
                                        hyper_degree[vc] += 1
        four_hyper, del_edges_four, num_four_hyper, num_tetra = four_way_hyper(hyper,  compute_dists, compute_hyper, num_classes, subsample_size, eps, norm, dists)
    return hyper + four_hyper, del_edges, del_edges_four, num_triangles, num_hyper, num_tetra, num_four_hyper, triangle_degree, hyper_degree


#======================FOUR WAY HYPER UTILS===========================
#sort the four vertices, with ordering 0,1,2,3
#input: the distance between the points
#output: (adj D) dot 1 and det D
def adj_D_1(dist_01, dist_02, dist_03, dist_12, dist_13, dist_23):
    u_sq = dist_01**2
    v_sq = dist_02**2
    w_sq = dist_03**2
    x_sq = dist_12**2
    y_sq = dist_13**2
    z_sq = dist_23**2

    uz_sq = u_sq*z_sq
    vy_sq = v_sq*y_sq
    wx_sq = w_sq*x_sq
    
    u_term = u_sq*(uz_sq - vy_sq - wx_sq)
    v_term = v_sq*(vy_sq - uz_sq - wx_sq)
    w_term = w_sq*(wx_sq - uz_sq - vy_sq)
    x_term = x_sq*(wx_sq - uz_sq - vy_sq)
    y_term = y_sq*(vy_sq - uz_sq - wx_sq)
    z_term = z_sq*(uz_sq - vy_sq - wx_sq)

    entry_0 = 2*x_sq*y_sq*z_sq + x_term + y_term + z_term
    entry_1 = 2*v_sq*w_sq*z_sq + v_term + w_term + z_term
    entry_2 = 2*u_sq*w_sq*y_sq + u_term + w_term + y_term
    entry_3 = 2*u_sq*v_sq*x_sq + u_term + v_term + x_term


    det = (u_sq**2)*(z_sq**2) + (v_sq**2)*(y_sq**2) + (w_sq**2)*(x_sq**2)
    det -= 2*u_sq*w_sq*x_sq*z_sq
    det -= 2*u_sq*v_sq*y_sq*z_sq
    det -= 2*v_sq*w_sq*x_sq*y_sq


    return ([entry_0, entry_1, entry_2, entry_3], det)


#input: (adj D) dot 1 and det D (output from adj_D_1)
#output: radius of the circumsphere
def circumsphere(alpha_p, det):
    r_sq =  det/(2*sum(alpha_p))
    return r_sq**0.5

def verify_four_hyper(a, b, c, d, compute_dists, dists, eps, norm, subsample_size, X=None):
    '''Check whether triangle ab, bc, ac is also a hyperedge
    If compute_dists is True, dists stores X_cs.  Otherwise dists
    is a dictionary with precomputed distances for each edge'''
    abcd = (a, b, c, d)
    print('tetrahedron', abcd)
    ab = (a, b)
    ac = (a, c)
    ad = (a, d)
    bc = (b, c)
    bd = (b, d)
    cd = (c, d)
    if norm == 'l2':
        dist_ab = dists[ab]
        dist_ac = dists[ac]
        dist_ad = dists[ad]
        dist_bc = dists[bc]
        dist_bd = dists[bd]
        dist_cd = dists[cd]
        alpha, det = adj_D_1(dist_ab, dist_ac, dist_ad, dist_bc, dist_bd, dist_cd)
        pos_idx = []
        #print('alpha', alpha)
        #print('det', det)
        #print('dists', dist_ab, dist_ac, dist_ad, dist_bc, dist_bd, dist_cd)
        for i, a in enumerate(alpha):
            if a/det > 0:
                pos_idx.append(i)
        #print('pos_idx', pos_idx)
        #assert(len(pos_idx) in [2,3,4])
        if len(pos_idx) == 4:
            r = circumsphere(alpha, det)
            return r <= eps
        elif len(pos_idx) == 2:
            if not compute_dists:
                return dists[abcd[pos_idx[0]], abcd[pos_idx[1]]] <= eps
            else:
                return get_dist(dists, abcd[pos_idx[0]], abcd[pos_idx[1]], subsample_size) <= eps
        else:
            return True # either 2 points have 0 dist or we already know all 3 combinations have r <= eps

    return True # otherwise it is linf, and all triangles in linf are also hyperedges

def four_way_hyper(three_way_hyper,  compute_dists, compute_hyper, num_classes, subsample_size, eps, norm, dists, Xs=None):
    three_way_hyper.sort()
    three_way_hyper_set = set(three_way_hyper)    
    hyper = [] # store hyperedges
    del_edges = set() # store redundant edge constraints
    num_tetra = 0
    num_hyper = 0

    if compute_hyper >= 4 and num_classes >= 4:
        # start hyperedge search
        # walk through each edge (a, b)
        for i, abc in enumerate(three_way_hyper):
            # walk through edges from b
            for j in range(i+1, len(three_way_hyper)): 
                other = three_way_hyper[j]
                
                # check if it is of the form abd.  If not then we must have already
                # passed through all entries of the form abd since our 3-hyper
                # are sorted
                if (other[0] != abc[0]) or (other[1] != abc[1]): break

                abd = (abc[0], abc[1], other[2])

                # check if acd and bcd exist
                acd = (abc[0], abc[2], abd[2])
                bcd = (abc[1], abc[2], abd[2])
                if acd not in three_way_hyper_set or bcd not in three_way_hyper_set: continue
                num_tetra += 1
                
                # check if it is a hyperedge
                if verify_four_hyper(abc[0], abc[1], abc[2], abd[2], compute_dists, dists, eps, norm, subsample_size, X=Xs):
                    num_hyper += 1
                    hyper.append((abc[0], abc[1], abc[2], abd[2])) 
                    del_edges.add(abc)
                    del_edges.add(abd)
                    del_edges.add(acd)
                    del_edges.add(bcd)
    return hyper, del_edges, num_hyper, num_tetra

        
# =====================PLOTTING FUNCT==============================
def plot_edge_density(density_plots_dir, subsample_size, classes_str, 
                    edge_degree, tri_degree_vwise, hyper_degree_vwise, X, binary=True):
    
    # visualize image with most edges, triangles, hyperedges
    max_edge_idx = np.argmax(edge_degree)
    c, i = get_class_idx(max_edge_idx, subsample_size)
    if binary:
        plt.imshow(X[max_edge_idx][0], cmap='gray')
    else:
        plt.imshow(X[max_edge_idx][0])
    plt.title('{} edges, class {}'.format(edge_degree[max_edge_idx], c))
    plt.savefig(os.path.join(density_plots_dir,str(subsample_size)+'_'+'_'.join(classes_str)+'_max_edge_degree'+str(max_edge_idx)+'.png'))
    plt.close()
    
    max_tri_idx = np.argmax(tri_degree_vwise)
    c, i = get_class_idx(max_tri_idx, subsample_size)
    if binary:
        plt.imshow(X[max_tri_idx][0], cmap='gray')
    else:
        plt.imshow(X[max_tri_idx][0])
    plt.title('{} triangles, class {}'.format(tri_degree_vwise[max_tri_idx], c))
    plt.savefig(os.path.join(density_plots_dir,str(subsample_size)+'_'+'_'.join(classes_str)+'_max_triangle_degree'+str(max_tri_idx)+'.png'))
    plt.close()
    
    max_hyper_idx = np.argmax(hyper_degree_vwise)
    c, i = get_class_idx(max_hyper_idx, subsample_size)
    if binary:
        plt.imshow(X[max_hyper_idx][0], cmap='gray')
    else:
        plt.imshow(X[max_hyper_idx][0])
    plt.title('{} hyperedges, class {}'.format(hyper_degree_vwise[max_hyper_idx], c))
    plt.savefig(os.path.join(density_plots_dir,str(subsample_size)+'_'+'_'.join(classes_str)+'_max_hyperedge_degree'+str(max_hyper_idx)+'.png'))
    plt.close()
    
    # remove vertices with 0 edges when plotting distribution
    edgesg0 = np.where(edge_degree > 0)[0]
    trianglesg0 = np.where(tri_degree_vwise > 0)[0]
    hyperg0 = np.where(hyper_degree_vwise > 0)[0]

    edge_degree = edge_degree[edgesg0]
    tri_degree_vwise = tri_degree_vwise[trianglesg0]
    hyper_degree_vwise = hyper_degree_vwise[hyperg0]

    #print('max edges =', max(edge_degree))
    #print('max triangles =', max(tri_degree_vwise))
    #print('max hyperedges =', max(hyper_degree_vwise))

    sns.distplot(edge_degree)
    plt.title("Distribution of number of edges per vertex")
    plt.xlabel("number of edges")
    plt.savefig(os.path.join(density_plots_dir,str(subsample_size)+'_'+'_'.join(classes_str)+'_edge_degree.pdf'))
    plt.close()

    sns.distplot(tri_degree_vwise)
    plt.title("Distribution of number of triangles per vertex")
    plt.xlabel("number of triangles")
    plt.savefig(os.path.join(density_plots_dir,str(subsample_size)+'_'+'_'.join(classes_str)+'_triangle_degree_v.pdf'))
    plt.close()

    sns.distplot(hyper_degree_vwise)
    plt.title("Distribution of number of hyperedges per vertex")
    plt.xlabel("number of hyperedges")
    plt.savefig(os.path.join(density_plots_dir,str(subsample_size)+'_'+'_'.join(classes_str)+'_hyperedge_degree_v.pdf'))
    plt.close()
