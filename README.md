# Layer-wised Model Aggregation for Personalized Federated Learning [[CVPR2022]](https://arxiv.org/abs/2205.03993)

**NOTE: This's not the official Repo.**

So maybe some hyperparameters not mentioned in paper aren't optimal.

All datasets mentioned in paper are supported(CIFAR10, ...).

If you find sth wrong about the code, feel free to open an issue or pr.

## Run

### Generating clients dataset

Segmenting by Dirichlet($\alpha$) and Randomly assigning classes are supported.

E.g.

```shell
python ./src/data/run.py --dataset cifar10 --classes 4 --client_num_in_total 10
```

Check `./src/data/run.py` for more Info of all arguments.



### Training

Scripts for running experiment with fixed args are in `./scripts`

```shell
cd ./scripts;
chmod +x ./*;
sh ${script}
```

Also, you can directly run `python ./src/server/pFedLA.py` with your custom arguments after generate clients dataset.


## Arguments

| Name          | Description                                                                                        |
| ------------- | -------------------------------------------------------------------------------------------------- |
| k             | Blocks retained by client in each round. Specifically for HeurpFedLA.                              |
| global_epochs | Communication rounds.                                                                              |
| local_epochs  | Client local training epochs.                                                                      |
| local_lr      | Client local optimizer's learning rate.                                                            |
| hn_lr         | Learning rate for each client's hypernetwork.                                                      |
| verbose_gap   | Logger report training results of selected clients after every `verbose_gap` communication rounds. |
| embedding_dim | Size of each client's embedding.                                                                   |
| hidden_dim    | Size of hidden layers in each client's hypernetwork.                                               |
| dataset       | Used dataset's name.                                                                               |
| batch_size    | Batch size for local training and test.                                                            |
| valset_ratio  | Ratio of validation set.                                                                           |
| testset_ratio | Ratio of test set.                                                                                 |
| gpu           | Set as non-zero value for using cuda.                                                              |
| log           | Set at non-zero value for saving log file in `./logs` .                                            |
| seed          | Random seed for running experiment.                                                                |
| save_period   | Temporarily save clients model parameters after every `save_period` communication rounds.          |

## TODO


Add some baselines for comparison

- [x] FedAvg
- FedBN
