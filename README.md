# csmcFederated-Learning (PyTorch)



## Data
* 

## Running the experiments
由于项目内容已修改，所以运行方式修改如下：

```
python federated_main.py --dataset mnist --model cnn --epochs 10 --num_users 100 --frac 0.1 --local_ep 5 --local_bs 32 --use_csmc 1 --compress_rank 10 --compress_layers fc1 --rates_mode random --rates_min 1e5 --rates_max 5e5

```

说明：

- `--use_csmc 1` 打开 csmcFL 行为（若设 0 则恢复传统 FedAvg 流程）。

  `--compress_rank` 控制保留奇异值数（论文的 λ̂）。

  `--compress_layers` 指定需要压缩的层（默认 `fc1`）。

  `--ro_search 1` 会在每轮对 candidate Ro 枚举并选择使估计上传时间最小的 Ro（与论文 Algorithm 1 思路类似，但为简化采用近似上传时间模型）。

# Non-IID 场景

```
python federated_main.py --dataset cifar  --iid 0 --epochs 200 --strategy csmcFL --compress_layers classifier.1 --local_ep 5 --local_bs 16 --gpu 0
    
```

```
参数
--dataset cifar	cifar	实验常用的数据集。
--iid 0	Non-IID	模拟实际场景下的数据异构性。
--epochs 200	经验值	提升到 200 轮以确保收敛（您代码中默认是 30 轮）。
--strategy csmcFL	csmcFL	启用联合优化方案。
--local_ep 5	E=5	本地训练周期（与 options.py 默认值一致）。
--local_bs 16	B=16	本地批次大小（与 options.py 默认值一致）。
--gpu 0	0	如果您有 GPU，使用 ID 0。如果没有，请将其改为 --gpu None 或省略。
```

#  IID (独立同分布) 场景

```
python federated_main.py \
    --dataset cifar \
    --iid 1 \
    --epochs 200 \
    --strategy csmcFL \
    --local_ep 5 \
    --local_bs 16 \
    --gpu 0
```



# 第一次仿真结果

```
小规模验证代码可行性
python federated_main.py --dataset cifar --iid 0 --epochs 2 --compress_layers classifier.1 --use_csmc 1 --gpu 0 --local_ep 1
```

```
Files already downloaded and verified
Files already downloaded and verified

=== 开始 csmcFL 联合优化 (交替迭代) ===
--- Joint Optimization Iteration 1 ---
   Updated Optimal Rank: 25 (Rate: 0.01)
   Updated Optimal X: Selected 1 Full / 29 Compressed
--- Joint Optimization Iteration 2 ---
   Updated Optimal Rank: 13 (Rate: 0.00)
   Updated Optimal X: Selected 1 Full / 29 Compressed
--- Joint Optimization Iteration 3 ---
   Updated Optimal Rank: 13 (Rate: 0.00)
   Updated Optimal X: Selected 1 Full / 29 Compressed

--- 联合优化在第 3 轮收敛，提前终止 ---
=== 联合优化结束 ===


=== 实验参数 ===
数据集: cifar | 模型: cnn | 分布: Non-IID
GT数量: 30 | 全局轮次: 2 | 本地轮次: 1
策略: csmcFL | 压缩层: classifier.1 | 压缩率: 0.003
================


=== 全局训练轮次 1/2 ===
| Global 0 | Local epoch 0 | batch 0 | loss 2.315732
| Global 0 | Local epoch 0 | batch 10 | loss 2.303655
| Global 0 | Local epoch 0 | batch 20 | loss 2.291454
| Global 0 | Local epoch 0 | batch 0 | loss 2.306019
| Global 0 | Local epoch 0 | batch 10 | loss 2.294977
| Global 0 | Local epoch 0 | batch 20 | loss 2.289475
| Global 0 | Local epoch 0 | batch 0 | loss 2.301861
| Global 0 | Local epoch 0 | batch 10 | loss 2.284396
| Global 0 | Local epoch 0 | batch 20 | loss 2.279735
| Global 0 | Local epoch 0 | batch 0 | loss 2.314758
| Global 0 | Local epoch 0 | batch 10 | loss 2.300656
| Global 0 | Local epoch 0 | batch 20 | loss 2.284026
| Global 0 | Local epoch 0 | batch 0 | loss 2.315598
| Global 0 | Local epoch 0 | batch 10 | loss 2.303637
| Global 0 | Local epoch 0 | batch 20 | loss 2.292131
| Global 0 | Local epoch 0 | batch 0 | loss 2.300267
| Global 0 | Local epoch 0 | batch 10 | loss 2.289670
| Global 0 | Local epoch 0 | batch 20 | loss 2.278166
| Global 0 | Local epoch 0 | batch 0 | loss 2.294576
| Global 0 | Local epoch 0 | batch 10 | loss 2.288960
| Global 0 | Local epoch 0 | batch 20 | loss 2.277692
| Global 0 | Local epoch 0 | batch 0 | loss 2.303719
| Global 0 | Local epoch 0 | batch 10 | loss 2.293044
| Global 0 | Local epoch 0 | batch 20 | loss 2.280101
| Global 0 | Local epoch 0 | batch 0 | loss 2.311305
| Global 0 | Local epoch 0 | batch 10 | loss 2.299774
| Global 0 | Local epoch 0 | batch 20 | loss 2.291063
| Global 0 | Local epoch 0 | batch 0 | loss 2.306428
| Global 0 | Local epoch 0 | batch 10 | loss 2.283913
| Global 0 | Local epoch 0 | batch 20 | loss 2.258105
| Global 0 | Local epoch 0 | batch 0 | loss 2.320552
| Global 0 | Local epoch 0 | batch 10 | loss 2.297325
| Global 0 | Local epoch 0 | batch 20 | loss 2.272908
| Global 0 | Local epoch 0 | batch 0 | loss 2.300180
| Global 0 | Local epoch 0 | batch 10 | loss 2.288576
| Global 0 | Local epoch 0 | batch 20 | loss 2.277374
| Global 0 | Local epoch 0 | batch 0 | loss 2.295579
| Global 0 | Local epoch 0 | batch 10 | loss 2.288806
| Global 0 | Local epoch 0 | batch 20 | loss 2.278401
| Global 0 | Local epoch 0 | batch 0 | loss 2.308307
| Global 0 | Local epoch 0 | batch 10 | loss 2.297481
| Global 0 | Local epoch 0 | batch 20 | loss 2.285363
| Global 0 | Local epoch 0 | batch 0 | loss 2.306956
| Global 0 | Local epoch 0 | batch 10 | loss 2.284011
| Global 0 | Local epoch 0 | batch 20 | loss 2.259118
| Global 0 | Local epoch 0 | batch 0 | loss 2.288710
| Global 0 | Local epoch 0 | batch 10 | loss 2.279200
| Global 0 | Local epoch 0 | batch 20 | loss 2.266847
| Global 0 | Local epoch 0 | batch 0 | loss 2.295845
| Global 0 | Local epoch 0 | batch 10 | loss 2.284260
| Global 0 | Local epoch 0 | batch 20 | loss 2.271958
| Global 0 | Local epoch 0 | batch 0 | loss 2.293873
| Global 0 | Local epoch 0 | batch 10 | loss 2.280890
| Global 0 | Local epoch 0 | batch 20 | loss 2.270185
| Global 0 | Local epoch 0 | batch 0 | loss 2.303050
| Global 0 | Local epoch 0 | batch 10 | loss 2.290194
| Global 0 | Local epoch 0 | batch 20 | loss 2.276428
| Global 0 | Local epoch 0 | batch 0 | loss 2.291843
| Global 0 | Local epoch 0 | batch 10 | loss 2.281426
| Global 0 | Local epoch 0 | batch 20 | loss 2.270897
| Global 0 | Local epoch 0 | batch 0 | loss 2.301835
| Global 0 | Local epoch 0 | batch 10 | loss 2.287779
| Global 0 | Local epoch 0 | batch 20 | loss 2.276280
| Global 0 | Local epoch 0 | batch 0 | loss 2.303435
| Global 0 | Local epoch 0 | batch 10 | loss 2.292238
| Global 0 | Local epoch 0 | batch 20 | loss 2.288026
| Global 0 | Local epoch 0 | batch 0 | loss 2.311984
| Global 0 | Local epoch 0 | batch 10 | loss 2.299728
| Global 0 | Local epoch 0 | batch 20 | loss 2.288656
| Global 0 | Local epoch 0 | batch 0 | loss 2.294303
| Global 0 | Local epoch 0 | batch 10 | loss 2.284320
| Global 0 | Local epoch 0 | batch 20 | loss 2.273115
| Global 0 | Local epoch 0 | batch 0 | loss 2.305949
| Global 0 | Local epoch 0 | batch 10 | loss 2.293591
| Global 0 | Local epoch 0 | batch 20 | loss 2.282019
| Global 0 | Local epoch 0 | batch 0 | loss 2.313552
| Global 0 | Local epoch 0 | batch 10 | loss 2.304207
| Global 0 | Local epoch 0 | batch 20 | loss 2.288754
| Global 0 | Local epoch 0 | batch 0 | loss 2.312542
| Global 0 | Local epoch 0 | batch 10 | loss 2.299932
| Global 0 | Local epoch 0 | batch 20 | loss 2.286725
| Global 0 | Local epoch 0 | batch 0 | loss 2.288800
| Global 0 | Local epoch 0 | batch 10 | loss 2.278454
| Global 0 | Local epoch 0 | batch 20 | loss 2.266216
| Global 0 | Local epoch 0 | batch 0 | loss 2.302006
| Global 0 | Local epoch 0 | batch 10 | loss 2.290238
| Global 0 | Local epoch 0 | batch 20 | loss 2.280007
| Global 0 | Local epoch 0 | batch 0 | loss 2.305887
| Global 0 | Local epoch 0 | batch 10 | loss 2.295340
| Global 0 | Local epoch 0 | batch 20 | loss 2.281329
=== 轮次 1 指标 ===
训练损失: 2.2889 | 测试准确率: 10.00%
模型一致性评分 (Similarity): 0.5557
本轮总传输量: 1252.97 Mbits
======================


=== 全局训练轮次 2/2 ===
| Global 1 | Local epoch 0 | batch 0 | loss 2.311709
| Global 1 | Local epoch 0 | batch 10 | loss 2.301634
| Global 1 | Local epoch 0 | batch 20 | loss 2.290458
| Global 1 | Local epoch 0 | batch 0 | loss 2.309711
| Global 1 | Local epoch 0 | batch 10 | loss 2.298003
| Global 1 | Local epoch 0 | batch 20 | loss 2.288879
| Global 1 | Local epoch 0 | batch 0 | loss 2.298387
| Global 1 | Local epoch 0 | batch 10 | loss 2.287304
| Global 1 | Local epoch 0 | batch 20 | loss 2.275570
| Global 1 | Local epoch 0 | batch 0 | loss 2.306878
| Global 1 | Local epoch 0 | batch 10 | loss 2.294235
| Global 1 | Local epoch 0 | batch 20 | loss 2.282908
| Global 1 | Local epoch 0 | batch 0 | loss 2.315154
| Global 1 | Local epoch 0 | batch 10 | loss 2.302255
| Global 1 | Local epoch 0 | batch 20 | loss 2.291240
| Global 1 | Local epoch 0 | batch 0 | loss 2.295813
| Global 1 | Local epoch 0 | batch 10 | loss 2.285774
| Global 1 | Local epoch 0 | batch 20 | loss 2.273224
| Global 1 | Local epoch 0 | batch 0 | loss 2.295766
| Global 1 | Local epoch 0 | batch 10 | loss 2.287217
| Global 1 | Local epoch 0 | batch 20 | loss 2.268045
| Global 1 | Local epoch 0 | batch 0 | loss 2.298952
| Global 1 | Local epoch 0 | batch 10 | loss 2.289580
| Global 1 | Local epoch 0 | batch 20 | loss 2.277527
| Global 1 | Local epoch 0 | batch 0 | loss 2.313974
| Global 1 | Local epoch 0 | batch 10 | loss 2.299782
| Global 1 | Local epoch 0 | batch 20 | loss 2.287978
| Global 1 | Local epoch 0 | batch 0 | loss 2.299577
| Global 1 | Local epoch 0 | batch 10 | loss 2.277433
| Global 1 | Local epoch 0 | batch 20 | loss 2.251632
| Global 1 | Local epoch 0 | batch 0 | loss 2.318291
| Global 1 | Local epoch 0 | batch 10 | loss 2.295204
| Global 1 | Local epoch 0 | batch 20 | loss 2.270548
| Global 1 | Local epoch 0 | batch 0 | loss 2.298488
| Global 1 | Local epoch 0 | batch 10 | loss 2.287571
| Global 1 | Local epoch 0 | batch 20 | loss 2.277104
| Global 1 | Local epoch 0 | batch 0 | loss 2.293936
| Global 1 | Local epoch 0 | batch 10 | loss 2.286226
| Global 1 | Local epoch 0 | batch 20 | loss 2.271282
| Global 1 | Local epoch 0 | batch 0 | loss 2.309058
| Global 1 | Local epoch 0 | batch 10 | loss 2.294698
| Global 1 | Local epoch 0 | batch 20 | loss 2.285254
| Global 1 | Local epoch 0 | batch 0 | loss 2.306773
| Global 1 | Local epoch 0 | batch 10 | loss 2.283610
| Global 1 | Local epoch 0 | batch 20 | loss 2.258395
| Global 1 | Local epoch 0 | batch 0 | loss 2.287045
| Global 1 | Local epoch 0 | batch 10 | loss 2.277432
| Global 1 | Local epoch 0 | batch 20 | loss 2.262221
| Global 1 | Local epoch 0 | batch 0 | loss 2.300639
| Global 1 | Local epoch 0 | batch 10 | loss 2.287830
| Global 1 | Local epoch 0 | batch 20 | loss 2.275445
| Global 1 | Local epoch 0 | batch 0 | loss 2.293092
| Global 1 | Local epoch 0 | batch 10 | loss 2.283301
| Global 1 | Local epoch 0 | batch 20 | loss 2.265682
| Global 1 | Local epoch 0 | batch 0 | loss 2.299559
| Global 1 | Local epoch 0 | batch 10 | loss 2.288969
| Global 1 | Local epoch 0 | batch 20 | loss 2.276443
| Global 1 | Local epoch 0 | batch 0 | loss 2.295332
| Global 1 | Local epoch 0 | batch 10 | loss 2.281816
| Global 1 | Local epoch 0 | batch 20 | loss 2.260019
| Global 1 | Local epoch 0 | batch 0 | loss 2.299748
| Global 1 | Local epoch 0 | batch 10 | loss 2.289235
| Global 1 | Local epoch 0 | batch 20 | loss 2.277583
| Global 1 | Local epoch 0 | batch 0 | loss 2.310233
| Global 1 | Local epoch 0 | batch 10 | loss 2.292642
| Global 1 | Local epoch 0 | batch 20 | loss 2.280314
| Global 1 | Local epoch 0 | batch 0 | loss 2.311350
| Global 1 | Local epoch 0 | batch 10 | loss 2.300271
| Global 1 | Local epoch 0 | batch 20 | loss 2.291040
| Global 1 | Local epoch 0 | batch 0 | loss 2.294474
| Global 1 | Local epoch 0 | batch 10 | loss 2.284733
| Global 1 | Local epoch 0 | batch 20 | loss 2.271369
| Global 1 | Local epoch 0 | batch 0 | loss 2.302620
| Global 1 | Local epoch 0 | batch 10 | loss 2.292171
| Global 1 | Local epoch 0 | batch 20 | loss 2.283432
| Global 1 | Local epoch 0 | batch 0 | loss 2.315757
| Global 1 | Local epoch 0 | batch 10 | loss 2.301674
| Global 1 | Local epoch 0 | batch 20 | loss 2.291383
| Global 1 | Local epoch 0 | batch 0 | loss 2.308442
| Global 1 | Local epoch 0 | batch 10 | loss 2.299189
| Global 1 | Local epoch 0 | batch 20 | loss 2.279903
| Global 1 | Local epoch 0 | batch 0 | loss 2.289050
| Global 1 | Local epoch 0 | batch 10 | loss 2.277009
| Global 1 | Local epoch 0 | batch 20 | loss 2.264671
| Global 1 | Local epoch 0 | batch 0 | loss 2.300565
| Global 1 | Local epoch 0 | batch 10 | loss 2.291282
| Global 1 | Local epoch 0 | batch 20 | loss 2.278880
| Global 1 | Local epoch 0 | batch 0 | loss 2.303040
| Global 1 | Local epoch 0 | batch 10 | loss 2.291928
| Global 1 | Local epoch 0 | batch 20 | loss 2.280525
=== 轮次 2 指标 ===
训练损失: 2.2873 | 测试准确率: 10.00%
模型一致性评分 (Similarity): 0.9796
本轮总传输量: 1252.97 Mbits
======================

最高测试准确率: 10.00%
预估平均单轮 FL 时间: 325.73 s
总运行时间: 2460.41 s

```

