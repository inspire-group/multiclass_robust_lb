from email.mime import base
from tkinter import W
import numpy as np
import argparse
import time
import os

from utils.data_utils import load_dataset_numpy, NumpyDataLoader

from sklearn.metrics import pairwise_distances

from scipy.sparse import coo_matrix

from cvxopt import solvers, matrix,  sparse, spmatrix
from hyper_utils import find_hyper, find_hyper_vwise, plot_edge_density
from solver_utils import construct_constraints, construct_incidence, run_solver, test_incidence


# ======================== DATA LOADING HELPERS =======================================
def get_dist_mat_name(args, class_idx1, class_idx2, subsample_size1, subsample_size2):
    '''Generate name for distance matrix'''
    if args.use_test:
        dist_mat_name = f'{args.dataset_in}_test_{args.classes[class_idx1]}_{args.classes[class_idx2]}_{subsample_size1}_{subsample_size2}_{args.norm}.npy'
    else:
        dist_mat_name = f'{args.dataset_in}_train_{args.classes[class_idx1]}_{args.classes[class_idx2]}_{subsample_size1}_{subsample_size2}_{args.norm}.npy'
    return dist_mat_name

def get_save_filename(args, subsample_size, classes_str = None, only_constraints=False):
    if args.use_all_classes:
        classes = 'all'
    else:
        classes = '_'.join(classes_str)
    split = 'train'
    if args.use_test:
        split = 'test'
    
    base_save_name = f'{args.dataset_in}_{split}_{classes}_{max(*subsample_size)}_{args.norm}'

    if args.balanced:
        base_save_name = base_save_name + '_balanced'
    base_save_name = base_save_name + '_hyper_' + str(args.compute_hyper)
    if args.run_integer:
        base_save_name += '_int'
    if only_constraints:
        return base_save_name + '_' + str(args.eps)
    else:
        return args.loss + '_' + base_save_name

def load_data(args, data_dir, batch_size, training_time):
    '''Load data into X_cs --> list of length n_classes, each element is a numpy array
       of the image data'''
    train_data, test_data, data_details = load_dataset_numpy(args, data_dir=data_dir,
                                                        training_time=training_time)
    if args.use_all_classes:
        args.n_classes = train_data.total_classes
        args.classes = list(range(args.n_classes))

    DATA_DIM = data_details['n_channels']*data_details['h_in']*data_details['w_in']

    if args.use_test:
        dataloader = NumpyDataLoader(test_data, DATA_DIM, batch_size=batch_size)
        num_samples = test_data.num_samples_per_class
    else:
        dataloader = NumpyDataLoader(train_data, DATA_DIM, batch_size=batch_size)
        num_samples = train_data.num_samples_per_class
    
    return dataloader, num_samples

def get_edge_pairs(args, dist_dir, dataloader, subsample_size):
    '''Obtain a list of edges and a dictionary of their corresponding distances'''
    edge_pairs = []
    with_edges = set()
    edge_degree = None
    dists = dataloader.dataset
    if args.plot_edge_density:
        edge_degree = np.zeros(sum(subsample_size))
    if not args.no_dist_hash:
        dists = {}
    if args.norm == 'l2':
        metric = 'euclidean'
    elif args.norm == 'linf':
        metric = 'chebyshev'
    else:
        raise ValueError('unsupported norm')

    class_start = np.cumsum(subsample_size)[:-1]
    class_start = [0] + list(class_start)
            
    for i in range(args.n_classes):
        for j in range(i+1, args.n_classes):
            dist_mat_name = get_dist_mat_name(args, i, j, subsample_size[i], subsample_size[j])
            if os.path.exists(dist_dir + '/' + dist_mat_name):
                D_ij = np.load(dist_dir + '/' + dist_mat_name)
            else:
                D_ij = np.zeros((subsample_size[i], subsample_size[j]))
                for batch_i_idx, batch_i in enumerate(dataloader.loader(i)):
                    for batch_j_idx, batch_j in enumerate(dataloader.loader(j)):
                        #print(batch_i_idx)
                        #print(batch_j_idx)
                        #rint(batch_i.shape)
                        #print(batch_j.shape)
                        D_ij[batch_i_idx * args.batch_size: (batch_i_idx + 1) * args.batch_size,
                             batch_j_idx * args.batch_size: (batch_j_idx + 1) * args.batch_size] =  pairwise_distances(batch_i,batch_j,metric=metric,n_jobs=-1)
                np.save(dist_dir + '/' + dist_mat_name, D_ij)
            edge_matrix = D_ij <= 2 * args.eps
            edges = np.where(edge_matrix==True)
            n_edges = len(edges[0])

            print('{} edges from class {} to class {}'.format(n_edges, str(args.classes[i]), str(args.classes[j])))
            
            for e in range(n_edges):
                j1=edges[0][e] + class_start[i]
                j2=edges[1][e]+ class_start[j]
                edge_pairs.append((j1, j2))
                if not args.no_dist_hash:
                    dists[(j1, j2)] = D_ij[edges[0][e], edges[1][e]]
                with_edges.add(j1)
                with_edges.add(j2)
                if args.plot_edge_density:
                    edge_degree[j1] += 1
                    edge_degree[j2] += 1
    num_edges = len(edge_pairs)
    return edge_pairs, num_edges, with_edges, dists, edge_degree


# ============================ MAIN FUNCTION ==============================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_in", default='MNIST', choices=['MNIST', 'CIFAR-10', 'CIFAR-100', 'CelebA'],
                        help="dataset to be used")
    parser.add_argument("--balanced", action='store_true', help='whether to enforced balanced num samples across each class')
    parser.add_argument("--norm", default='l2',
                        help="norm to be used")
    parser.add_argument('--num_samples', type=int, default=5000,
                        help='maximum number of samples per class to compute lower bound on.')
    parser.add_argument('--mosek', action='store_true',
                        help='use mosek solver')
    parser.add_argument('--eps', type=float, default=None,
                        help='lp ball radius')
    parser.add_argument('--use_test', dest='use_test', action='store_true',
                        help='compute lower bound on test set')
    parser.add_argument('--batch_size', type=int, default=1000)
    parser.add_argument('--run_generic', dest='run_generic', action='store_true',
                        help='use generic solver')
    parser.add_argument('--run_integer', action='store_true', help='run integer programming')
    parser.add_argument('--run_nonconvex', action='store_true', help='run generic nonconvex solver')
    parser.add_argument('--new_marking_strat', type=str, default=None)
    parser.add_argument('--num_reps', type=int, default=1)
    parser.add_argument('--remove_redundant', action='store_true',
                        help='remove redundant constraints')
    parser.add_argument('--out_dir', type=str, default='/scratch/gpfs/sihuid/',
                        help='path to directory for output logs')
    parser.add_argument('--data_dir', type=str, default='/scratch/gpfs/sihuid/data',
                        help='path to data directory')
    parser.add_argument('--compute_hyper', type=int, default=2, choices=[2,3,4],
                        help='compute n-way hyperedges and use n-way constraints')
    parser.add_argument('--save_constraints', action='store_true',
                        help='whether to save the edge constraints to file')
    parser.add_argument('--plot_edge_density', action='store_true',
                        help='plot edge densities')
    parser.add_argument('--no_dist_hash', action='store_true',
                        help='does not store dictionary of computed distances')
    parser.add_argument('--hyper_vwise', action='store_true',
                        help='run vertex-wise iteration version of hyperedge computation')
    parser.add_argument('--classes', nargs='+', help='classes to use', default=None)
    parser.add_argument('--use_all_classes', action='store_true', 
                        help='whether to use all classes of dataset, if set to true, classes argument is ignored')
    parser.add_argument("--loss", default='0-1', choices=['0-1', 'ce'],
                        help='loss to use')

    args = parser.parse_args()
    if args.classes is not None:
        classes_str = args.classes
        if 'all' not in args.classes:
            args.classes = [int(c) for c in args.classes]
            args.classes = sorted(args.classes)
            args.n_classes = len(args.classes)
        else:
            args.classes.remove('all')
            args.classes = [int(c) for c in args.classes]
            args.classes = sorted(args.classes)
            args.classes += ['all']
            args.n_classes = len(args.classes)
        print(args.classes)
    else:
        args.n_classes = None # we'll set this after the dataset is loaded
        classes_str = None

    # load data
    t1 = time.process_time()
    dataloader, num_samples = load_data(args, data_dir=args.data_dir, batch_size=args.batch_size, training_time=False)
    print('num samples', num_samples)
    t2 = time.process_time()
    print('Time taken to load data', t2 - t1)
    
    # set up output paths
    dist_dir = os.path.join(args.out_dir, 'distances')
    cost_res_eps_dir = os.path.join(args.out_dir, 'cost_results', args.dataset_in, str(args.eps))
    timing_res_dir = os.path.join(cost_res_eps_dir, 'timing_results')
    opt_probs_dir = os.path.join(args.out_dir, 'graph_data', 'optimal_probs')
    opt_constraints_dir = os.path.join(args.out_dir, 'graph_data', 'constraints')
    density_plots_dir = os.path.join(args.out_dir, 'density_plots', args.dataset_in, str(args.eps))

    if not os.path.exists(dist_dir):
        os.makedirs(dist_dir)
    if not os.path.exists(timing_res_dir):
        os.makedirs(timing_res_dir)
    if not os.path.exists(opt_probs_dir):
        os.makedirs(opt_probs_dir)
    if not os.path.exists(opt_constraints_dir):
        os.makedirs(opt_constraints_dir)
    if not os.path.exists(density_plots_dir):
        os.makedirs(density_plots_dir)

        
    for rep in range(args.num_reps):

        # extract edges and their dists from data
        t1 = time.process_time()
        edge_pairs, num_edges, with_edges, dists, edge_degree = get_edge_pairs(args, dist_dir, dataloader, num_samples)
        t2 = time.process_time()
        print('Time taken to get edge pairs', t2 - t1)
        
        print('Total num edges: {}'.format(num_edges))

        # compute hyperedges
        t1 = time.process_time()
        num_edges_updated = num_edges
        if args.hyper_vwise:
            hyper, four_hyper, del_edges, del_edges_four, num_triangles, num_hyper, num_tetra, num_four_hyper, tri_degree_vwise, \
                hyper_degree_vwise = find_hyper_vwise(edge_pairs, args.no_dist_hash, args.compute_hyper, args.n_classes, num_samples, args.eps, args.norm,
                dists=dists, plot_edges=args.plot_edge_density)
        else:
            hyper, four_hyper, del_edges, del_edges_four, num_triangles, num_hyper, num_tetra, num_four_hyper, tri_degree_vwise, \
                hyper_degree_vwise = find_hyper(edge_pairs, args.no_dist_hash, args.compute_hyper, args.n_classes, num_samples, args.eps, args.norm,
                dists=dists, plot_edges=args.plot_edge_density, Xs=dataloader.dataset)
        t2 = time.process_time()
        print('Time taken to get compute hyperedges overall', t2 - t1)
        del dists
        num_hyper_updated = num_hyper

        if args.plot_edge_density:
            plot_edge_density(density_plots_dir, num_samples, classes_str, 
                edge_degree, tri_degree_vwise, hyper_degree_vwise, dataloader.dataset,
                    ('MNIST' in args.dataset_in) )
            del edge_degree
            del tri_degree_vwise
            del hyper_degree_vwise

        assert(num_hyper == len(hyper))
        print("number of triangles:", num_triangles)
        print("number of 3-way hyperedges:", num_hyper)
        print("number of tetrahedrons:", num_tetra)
        print("number of 4-way hyperedges", num_four_hyper)
        if args.remove_redundant:
            num_edges_updated -= len(del_edges)
            num_hyper_updated -= len(del_edges_four)
            
        else:
            del_edges = set()
            del_edges_four = set()

        if args.run_generic:
            # construct incidence matrices (sparse representation)
            t1 = time.process_time()
            incidence_row_coord, incidence_col_coord = construct_incidence(num_edges_updated, num_hyper_updated, hyper, num_four_hyper, four_hyper, edge_pairs, del_edges, del_edges_four)
            # assuming edges only
            even_arr = incidence_col_coord[0::2]
            odd_arr = incidence_col_coord[1::2]
            full_row_coord = np.concatenate([even_arr, odd_arr])
            full_col_coord = np.concatenate([odd_arr, even_arr])
            t2 = time.process_time()
            print('Time taken to construct incidence', t2 - t1)
            del hyper
            del edge_pairs
            del del_edges
            
            # add additional <= 1 constraint for isolated vertices
            n_rows = num_edges_updated + num_hyper_updated + num_four_hyper
            v = sum(num_samples)
            t1 = time.process_time()
            
            # add q < 1 constraints (if not running geometric program)
            if args.loss == '0-1' or args.run_nonconvex:
                incidence_row_coord, incidence_col_coord = construct_constraints(v, n_rows, incidence_row_coord, incidence_col_coord,
                                                                            args.remove_redundant, with_edges)
            if args.save_constraints:
                save_file_name = get_save_filename(args, num_samples, classes_str=classes_str, only_constraints=True)
                np.save(opt_constraints_dir + '/' + save_file_name + '_row_coord', incidence_row_coord)
                np.save(opt_constraints_dir + '/' + save_file_name + '_col_coord', incidence_col_coord)
            t2 = time.process_time()
            print('Time taken to add identity to incidence', t2 - t1)
            
            # open files for writing output
            save_file_name = get_save_filename(args, num_samples, classes_str=classes_str, only_constraints=False)

            f = open(cost_res_eps_dir + '/' + save_file_name + '.txt', 'a')
            f.write('num edges: {}\n'.format(num_edges))
            f.write('num triangles: {}\n'.format(num_triangles))
            f.write('num 3-way hyperedges: {}\n'.format(num_hyper))
            f.write('num tetrahedrons: {}\n'.format(num_tetra))
            f.write('num 4-way hyperedges: {}\n'.format(num_four_hyper))
            if args.loss == '0-1' or args.run_nonconvex:
                f.write('num_constraints: {}\n'.format(n_rows))
            f_time = open(timing_res_dir + '/' + save_file_name + '.txt', 'a')

            # solve optimization
            time3=time.process_time()
            output = run_solver(incidence_row_coord, incidence_col_coord, args.loss, v, n_rows, args.mosek, args.run_nonconvex, args.run_integer)
            time4=time.process_time()

            if not args.run_integer:
                status = output['status']
                sol = output['x']
            else:
                status, sol = output

            if status == 'optimal':
                f_time.write('Rep: {} \t Runtime: {}'.format(rep, time4 - time3))
                print('Runtime: {}'.format(time4 - time3))
            else:
                print('Optimal not found!')
                f_time.write('Rep: {} \t Runtime: {}, OPTIMAL NOT FOUND'.format(rep, time4 - time3))
           
            # use incidence_row_coord, incidence_col_coord without hyper compute
            
            # len(full_row_coord) = n_rows
            A = coo_matrix(([1]*len(full_row_coord), (full_row_coord, full_col_coord)), shape=(v, v)).toarray()
            # (1^T x)^2 / (x^T (A+I) x)
            numerator = sum(sol)**2
            denominator = np.matmul(np.matmul(sol.T, A+np.identity(v)), sol).item()
            independent_set = numerator/denominator
            print("lower bound on independence number:", independent_set)
            print("lower bound on loss of deterministic classifier", 1 - (independent_set / v))
            # log output

            if not args.run_integer:
                f.write('Rep {}, primal objective: {}\n'.format(rep, output['primal objective']))
                print('Rep {}, primal objective: {}\n'.format(rep, output['primal objective']))
            if args.loss == '0-1':
                print("optimal 0-1 loss:", 1-sum(np.array(sol)*1/v)[0])
                f.write("Rep {}, optimal 0-1 loss: {}".format(rep, 1-sum(np.array(sol)*1/v)[0]))
            
            if args.loss == '0-1' or args.run_nonconvex:
                np.save(opt_probs_dir + '/' + save_file_name + '_eps_{}'.format(args.eps), np.array(sol))
            else: 
                # using geometric program, we obtain x = log q, so we need to exponentiate to obtain q
                np.save(opt_probs_dir + '/' + save_file_name + '_eps_{}'.format(args.eps), np.array(np.exp(sol)))
            
            # close log files
            f.close()
            f_time.close()


if __name__ == '__main__':
    main()

    # tests for incidence correctness
    #test_incidence([(0,1),(0,2),(1,2)], [], 3) 
    #test_incidence([], [(0,1,2)], 3)
    #test_incidence([(0,1),(0,2),(0,3),(0,4),(1,2),(3,4)], [], 5) 
    #test_incidence([(0,1),(0,2),(1,2)], [(0,3,4)], 5) 
    #test_incidence([], [(0,1,2),(0,3,4)], 5)
    #test_incidence([(0,1),(0,2),(0,3),(0,4),(1,2),(1,3),(1,4)], [], 5)
    #test_incidence([], [(0,1,2),(0,1,3),(0,1,4)], 5)

    
