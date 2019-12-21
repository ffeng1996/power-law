## train a dense cnn model for mnist
#python mnist_dense.py
## prune 70% connections
#python mnist_sparse.py --num_epochs=200 --prune_fraction=0.7
# plot the ccdf plot
python degree_plot_slope_clauset_discrete.py --option="ccdf"

## train a dense network for cifar
#python cifar_dense.py
## prune 70% connections
#python cifar_sparse.py --num_epochs=100 --prune_fraction=0.7
# plot the ccdf plot
python degree_plot_slope_cifar_clauset_discrete.py --option="ccdf"
