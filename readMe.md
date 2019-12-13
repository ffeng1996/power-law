## Requirements
Recommend install in virtual environment
```
$ conda create -n yourenvname python=2.7 anaconda
```

Activate the enviorment
```
conda activate yourenvname
```
Install the required the packages inside the virtual environment
```
sh installation.sh
```
## Run the models

For experiments in section **4.2**

```
bash power-law/expt2/fc/run.sh
```
For experiments in section **5.1**

```
bash power-law/expt3/fc/run_pathnet.sh
```
For mlp on mnist in section **5.2**:

```
bash power-law/expt3/fc/run_pnn.sh
```
For cnn on cifar in section **5.2**:
```
bash power-law/expt3/cnn/run_cifar_pnn.sh
```

## Dataset

- MNIST is inside this folder
- CIFAR dataset need to be prepared using pylearn2 following https://github.com/MatthieuCourbariaux/BinaryConnect
