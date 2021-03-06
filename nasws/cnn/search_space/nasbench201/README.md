# An Algorithm-Agnostic NAS Benchmark (AA-NAS-Bench)
**Direct adaptation from DXY's code**

We propose an Algorithm-Agnostic NAS Benchmark (AA-NAS-Bench) with a fixed search space, which provides a unified benchmark for almost any up-to-date NAS algorithms.
The design of our search space is inspired from that used in the most popular cell-based searching algorithms, where a cell is represented as a directed acyclic graph. Each edge here is associated with an operation selected from a predefined operation set. For it to be applicable for all NAS algorithms, the search space defined in AA-NAS-Bench includes 4 nodes and 5 associated operation options, which generates 15,625 neural cell candidates in total.

In this Markdown file, we provide:
- Detailed instruction to reproduce AA-NAS-Bench.
- 10 NAS algorithms evaluated in our paper.

Note: please use `PyTorch >= 1.1.0` and `Python >= 3.6.0`.

## How to Use AA-NAS-Bench

1. Creating AA-NAS-Bench API from a file:
```
from aa_nas_api import AANASBenchAPI
api = AANASBenchAPI('$path_to_meta_aa_nasbench101_file')
api = AANASBenchAPI('AA-NAS-Bench-v1_0.pth')
```

2. Show the number of architectures `len(api)` and each architecture `api[i]`:
```
num = len(api)
for i, arch_str in enumerate(api):
  print ('{:5d}/{:5d} : {:}'.format(i, len(api), arch_str))
```

3. Show the results of all trials for a single architecture:
```
# show all information for a specific architecture
api.show(1)
api.show(2)

# show the mean loss and accuracy of an architecture
info = api.query_meta_info_by_index(1)
loss, accuracy = info.get_metrics('cifar10', 'train')
flops, params, latency = info.get_comput_costs('cifar100')

# get the detailed information
results = api.query_by_index(1, 'cifar100')
print ('There are {:} trials for this architecture [{:}] on cifar100'.format(len(results), api[1]))
print ('Latency : {:}'.format(results[0].get_latency()))
print ('Train Info : {:}'.format(results[0].get_train()))
print ('Valid Info : {:}'.format(results[0].get_eval('x-valid')))
print ('Test  Info : {:}'.format(results[0].get_eval('x-test')))
# for the metric after a specific epoch
print ('Train Info [10-th epoch] : {:}'.format(results[0].get_train(10)))
```

4. Query the index of an architecture by string
```
index = api.query_index_by_arch('|nor_conv_3x3~0|+|nor_conv_3x3~0|avg_pool_3x3~1|+|skip_connect~0|nor_conv_3x3~1|skip_connect~2|')
api.show(index)
```

5. For other usages, please see `lib/aa_nas_api/api.py`

## Instruction to Generate AA-NAS-Bench

1. generate the meta file for AA-NAS-Bench using the following script, where `AA-NAS-BENCH` indicates the name and `4` indicates the maximum number of nodes in a cell.
```
bash scripts-search/AA-NAS-meta-gen.sh AA-NAS-BENCH 4
```

2. train earch architecture on a single GPU (see commands in `output/AA-NAS-BENCH-4/meta-node-4.opt-script.txt` which is automatically generated by step-1).
```
CUDA_VISIBLE_DEVICES=0 bash ./scripts-search/AA-NAS-train-archs.sh     0   389 -1 '777 888 999'
```
This command will train 390 architectures (id from 0 to 389) using the following four kinds of splits with three random seeds (777, 888, 999).

|     Dataset     |     Train     | Eval  |
|:---------------:|:-------------:|:-----:|
| CIFAR-10        | train         | valid |
| CIFAR-10        | train + valid | test  |
| CIFAR-100       | train         | valid+test |
| ImageNet-16-120 | train         | valid+test |

3. calculate the latency, merge the results of all architectures, and simplify the results.
(see commands in `output/AA-NAS-BENCH-4/meta-node-4.cal-script.txt` which is automatically generated by step-1).
```
OMP_NUM_THREADS=6 CUDA_VISIBLE_DEVICES=0 python exps/AA-NAS-statistics.py --mode cal --target_dir 000000-000389-C16-N5
```

4. merge all results into a single file for AA-NAS-Bench-API.
```
OMP_NUM_THREADS=4 python exps/AA-NAS-statistics.py --mode merge
```

[option] train a single architecture on a single GPU.
```
CUDA_VISIBLE_DEVICES=0 bash ./scripts-search/AA-NAS-train-net.sh resnet 16 5
CUDA_VISIBLE_DEVICES=0 bash ./scripts-search/AA-NAS-train-net.sh '|nor_conv_3x3~0|+|nor_conv_3x3~0|nor_conv_3x3~1|+|skip_connect~0|skip_connect~1|skip_connect~2|' 16 5
```

## To Reproduce 10 Baseline NAS Algorithms in AA-NAS-Bench

We have tried our best to implement each method. However, still, some algorithms might obtain non-optimal results since their hyper-parameters might not fit our AA-NAS-Bench.
If researchers can provide better results with different hyper-parameters, we are happy to update results according to the new experimental results. We also welcome more NAS algorithms to test on our dataset and would include them accordingly.

- [1] `CUDA_VISIBLE_DEVICES=0 bash ./scripts-search/algos/DARTS-V1.sh cifar10 -1`
- [2] `CUDA_VISIBLE_DEVICES=0 bash ./scripts-search/algos/DARTS-V2.sh cifar10 -1`
- [3] `CUDA_VISIBLE_DEVICES=0 bash ./scripts-search/algos/GDAS.sh     cifar10 -1`
- [4] `CUDA_VISIBLE_DEVICES=0 bash ./scripts-search/algos/SETN.sh     cifar10 -1`
- [5] `CUDA_VISIBLE_DEVICES=0 bash ./scripts-search/algos/ENAS.sh     cifar10 -1`
- [6] `CUDA_VISIBLE_DEVICES=0 bash ./scripts-search/algos/RANDOM-NAS.sh cifar10 -1`
- [7] `bash ./scripts-search/algos/R-EA.sh -1`
- [8] `bash ./scripts-search/algos/Random.sh -1`
- [9] `bash ./scripts-search/algos/REINFORCE.sh -1`
- [10] `bash ./scripts-search/algos/BOHB.sh -1`
