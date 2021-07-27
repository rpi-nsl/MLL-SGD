### Multi-Level Local SGD simulation code

Corresponding paper:
```
    T. Castiglia, A. Das, and S. Patterson “Multi-Level Local SGD: Distributed SGD for Heterogeneous Hierarchical Networks” in ICLR, 2021
```
If you use the code base or refer to our paper, please use the following bibtex entry:
```
@inproceedings{
castiglia2021multilevel,
title={Multi-Level Local {SGD}: Distributed {SGD} for Heterogeneous Hierarchical Networks},
author={Timothy Castiglia and Anirban Das and Stacy Patterson},
booktitle={International Conference on Learning Representations},
year={2021},
url={https://openreview.net/forum?id=C70cp4Cn32}
}
```

---
This code is for simulating a multi-level network with
heterogeneous workers, and running MLL-SGD to train a model
on a set of data. The code is NOT for deployment purposes, and there
is much room for optimization.

One can install our environment with all the dependencies with Anaconda:
```shell
conda create env -f flearn.yml 
```

Figure results seen in the paper are under the "images" folder.
Raw data results can be found under the "results" folder.

To rerun our experiments, run:
```shell
./run_exps.sh
```
The bash script currently runs all experiments
sequentially. They can be run in parallel depending on your
machine's available memory or available gpus.

To plot the results:
```shell
    python plot_exps.py
```

One can also run MLL-SGD with your own parameters.
mll_sgd.py help output with extra comments:

Usage: 
```shell
mll_sgd.py [-h] [--data [DATA]] [--model [MODEL]]
                  [--hubs [HUBS]] [--workers [WORKERS]] [--tau [TAU]]
                  [--q [Q]] [--graph [GRAPH]] [--epochs [EPOCHS]]
                  [--batch [BATCH]] [--prob [PROB]] [--fed [FED]]
                  [--chance [CHANCE]]
```

Run Multi-Level Local SGD.

Following are some optional arguments:
```shell
  -h, --help           show this help message and exit
  --data [DATA]        dataset to use in training.
                           Value of 0 = MNIST data
                           Value of 1 = EMNIST data
                           Value of 2 = CIFAR-10 data
  --model [MODEL]      model to use in training.
                           Value of 0 = Logistic regression 
                           Value of 1 = CNN model for EMNIST
                           Value of 2 = CIFARNet CNN model
                           Value of 3 = ResNet-18 model
  --hubs [HUBS]        number of hubs in system.
  --workers [WORKERS]  number of workers per hub.
  --tau [TAU]          number of local iterations for worker.
  --q [Q]              number of sub-network iterations before global
                       averaging.
  --graph [GRAPH]      graph file ID to use for hub network.
                           Values 1-4 use graphs in the "graphs" folder
                           Value of 5 uses complete graph
                           Value of 6 uses a line graph
  --epochs [EPOCHS]    Number of epochs/global iterations to train for.
  --batch [BATCH]      Batch size to use in Mini-batch SGD.
  --prob [PROB]        Indicates with probability distribution to use for
                       workers.
                           Value of 0 = All worker probabilities are 1 
                           Value of 1 = Use fixed probability defined by "chance" input
                           Value of 2 = Uniform probability distribution from 0.1 to 1
                           Value of 3 = 10% of workers with probability 0.1, rest with 0.6
                           Value of 4 = 10% of workers with probability 1, rest with 0.5
                           Value of 5 = 10% of workers with probability 0.6, rest with 0.9
  --fed [FED]          Indicates if worker sets should be different sizes.
                           False = All workers are given equal sized datasets
                           True  = Sub-networks receive either 5%, 10%, 20%, 25%, or 40% of the total dataset
  --chance [CHANCE]    Fixed probability of taking gradient step.
                           Only active when prob = 1.
```