# Characterizing the Optimal 0 − 1 Loss for Multi-class Classification with a Test-time Attacker

## Compute optimal loss of existing datasets

The main document to compute optimal loss of a dataset is **`optimal_log_loss_lp_hyper.py`**. It is able to compute loss on dataset `MNIST`, `CIFAR-10`, `CIFAR-100` and `CelebA` (using `--dataset_in` argument) and `0-1` or cross-entropy loss (using `--loss` argument). We can compute up to 4-way hyperedges. Users can also select specific classes of a dataset for computation (using `--classes` argument). More arguments and flags can be found under `main()`.

### Compute full dataset

For example, to compute optimal `0-1` loss of full `CIFAR-10` dataset at $\epsilon=3$ up to degree 3 hyperedges, run the following command (substituting directory path):

`python3 optimal_log_loss_lp_hyper.py --remove_redundant --out_dir OUT_DIR --data_dir DATA_DIR --compute_hyper 3 --norm l2 --dataset_in CIFAR-10 --loss 0-1 --use_all_classes --eps 3  --run_generic --run_nonconvex --num_samples 8000 --mosek`.

The optimal loss along with edge information will be saved in a file under `cost_result` folder. Note that in order to include all data, `--num_samples` $\ge \max$ sample per class.

### Compute partial dataset

To compute the optimal `0-1` loss of `MNIST` class 1, 4 and 7 at $\epsilon=3$ up to degree 2 hyperedges with 1000 samples per class, run

`python3 optimal_log_loss_lp_hyper.py --remove_redundant --out_dir OUT_DIR --data_dir DATA_DIR --norm l2 --dataset_in MNIST --loss 0-1 --classes 1 4 7 --eps 3  --run_generic --run_nonconvex --num_samples 1000 --mosek`.
