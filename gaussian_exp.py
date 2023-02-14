import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import pairwise_distances
from optimal_log_loss_lp_hyper import find_hyper, construct_constraints, construct_incidence, run_solver
import argparse


def center_locations(dist_from_origin):
    x_1 = -dist_from_origin / 2 * np.sqrt(3)
    y_1 = dist_from_origin / 2
    
    x_2 = -x_1
    y_2 = y_1
    
    x_3 = 0
    y_3 = -dist_from_origin
    
    return (x_1, y_1), (x_2, y_2), (x_3, y_3)

def get_samples(centers, var, n):
    samples_per_class = []
    for c_x, c_y in centers:
        samples = np.sqrt(var) * np.random.randn(n, 2)
        samples[:, 0] = samples[:, 0] + c_x
        samples[:, 1] = samples[:, 1] + c_y
        samples_per_class.append(samples)
    return samples_per_class

def get_edge_pairs(samples, eps):
    edge_pairs = []
    with_edges = set()
    dists = {}
    
    for i in range(len(samples)):
        for j in range(i+1, len(samples)):
            D_ij = pairwise_distances(samples[i],samples[j],metric='euclidean')
            edge_matrix = D_ij <= 2 * eps
            edges = np.where(edge_matrix==True)
            n_edges = len(edges[0])
            for e in range(n_edges):
                j1=edges[0][e] + len(samples[0]) * i
                j2=edges[1][e]+ len(samples[0]) * j
                edge_pairs.append((j1, j2))
                dists[(j1, j2)] = D_ij[edges[0][e], edges[1][e]]
                with_edges.add(j1)
                with_edges.add(j2)
    return edge_pairs, with_edges, dists

def error_via_classifier(sample1, sample2, sample3, eps):    
    misclassified1 = np.sum(np.logical_or(sample1[:, 0] > -eps, (sample1[:,1] - 1/np.sqrt(3) * sample1[:, 0]) < (2 / np.sqrt(3) * eps)))
    misclassified2 = np.sum(np.logical_or(sample2[:, 0] < eps, (sample2[:,1] + 1/np.sqrt(3) * sample2[:, 0]) < (2 / np.sqrt(3) * eps)))
    cond1 = np.logical_and(sample3[:,0] < 0, sample3[:,1] - 1/np.sqrt(3) * sample3[:,0] > -2/np.sqrt(3) * eps)
    cond2 = np.logical_and(sample3[:,0] > 0, sample3[:,1] + 1/np.sqrt(3) * sample3[:,0] > -2/np.sqrt(3) * eps)
    misclassified3 = np.sum(np.logical_or(cond1, cond2))

    return (misclassified1 + misclassified2 + misclassified3) / (len(sample1) + len(sample2) + len(sample3))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--variance', type=float, default=1.0)
    parser.add_argument('--dist_from_origin', type=float, default=3.0)
    parser.add_argument('--num_samples', type=int, default=10000)
    parser.add_argument('--eps', type=float, default=0.5)
    args = parser.parse_args()
    
    dist_from_origin = args.dist_from_origin
    var = args.variance
    n = args.num_samples
    eps = args.eps

    centers = center_locations(dist_from_origin)
    sample_1, sample_2, sample_3 = get_samples(centers, var, n)
    #np.save('gauss_samples/gauss_dist_{}_var_{}_n_{}_sample_1.npz'.format(dist_from_origin, var, n), sample_1)
    #np.save('gauss_samples/gauss_dist_{}_var_{}_n_{}_sample_2.npz'.format(dist_from_origin, var, n), sample_2)
    #np.save('gauss_samples/gauss_dist_{}_var_{}_n_{}_sample_3.npz'.format(dist_from_origin, var, n), sample_3)
    err_classifier = error_via_classifier(sample_1, sample_2, sample_3, eps)
    edges, with_edges, dists = get_edge_pairs([sample_1, sample_2, sample_3], eps)
    num_edges = len(edges)
    hyper, _, del_edges, _, num_triangles, num_hyper, _, _, _, _ = find_hyper(edges, False, 3, 3, [n, n, n], eps, 'l2',
                dists=dists, plot_edges=False)

    num_edges_updated = num_edges - len(del_edges)
    incidence_row_coord, incidence_col_coord = construct_incidence(num_edges_updated, num_hyper, hyper, 0, [], edges, del_edges, {})
    incidence_row_coord, incidence_col_coord = construct_constraints(n*3, num_edges_updated + num_hyper, incidence_row_coord, incidence_col_coord,
                                                                            True, with_edges)
    
    print('constraint size hyper', n*3, num_edges_updated + num_hyper)
    output_hyper = run_solver(incidence_row_coord, incidence_col_coord, '0-1', n*3, num_edges_updated + num_hyper, True, False, False)

    num_edges_updated = num_edges
    incidence_row_coord, incidence_col_coord = construct_incidence(num_edges_updated, 0, [], 0, [], edges, {}, {})
    incidence_row_coord, incidence_col_coord = construct_constraints(n*3, num_edges_updated, incidence_row_coord, incidence_col_coord,
                                                                            True, with_edges)
    
    print('constraint size', n*3, num_edges_updated)
    output = run_solver(incidence_row_coord, incidence_col_coord, '0-1', n*3, num_edges_updated + num_hyper, True, False, False)

    #output_int = run_solver(incidence_row_coord, incidence_col_coord, '0-1', n*3, num_edges_updated + num_hyper, True, False, True)

    #np.save('gauss_samples/gauss_dist_{}_var_{}_n_{}_weights.npz'.format(dist_from_origin, var, n), np.array(output['x']))
    print("num edges:", num_edges)
    print("num hyper:", num_hyper)
    print("estimated error via classifier:", err_classifier)
    print("optimal 0-1 loss w/o hyper:", (1-np.array(output['x']).sum()/ (n*3)))
    print("optimal 0-1 loss w hyper:", (1-np.array(output_hyper['x']).sum()/ (n*3)))
    #print("optimal 0-1 loss IP", (1-np.array(output_int[1]).sum() / (n*3)))

if __name__ == "__main__":
    main()
