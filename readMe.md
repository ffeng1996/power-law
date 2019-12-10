#Requirements
Recommend install in virtual environment
```
$ conda create -n yourenvname anaconda
```

Activate the enviorment
```
conda activate yourenvname
```
Install the required the packages inside the virtual environment
```
./installation.sh
```
#Run the models

For experiments in section 4.2

```
power-law/expts2/fc/run.sh
```
For experiments in section 5.1

```
power-law/expts3/fc/run_pathnet.sh
```
For mlp on mnist in section 5.2:

```
power-law/expts3/fc/run_pnn.sh
```
For cnn on cifar in section 5.2:
```
power-law/expts3/cnn/run_cifar_pnn.sh
```

