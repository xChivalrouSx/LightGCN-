# LightGCN+
This is a tensorflow implementation for my Master of Science Thesis:

Master of Science Thesis Paper: 
---
[Eczaneler İçin Kullanıcı Tabanlı Müstahzar Öneri Sistemi (User-Based Preparation/Drug Recommendation System for Pharmacies) - 2022](https://github.com/xChivalrouSx/LightGCN-plus/blob/main/paper/mertcakar-master-thesis-lightgcnplus.pdf)

## Introduction
In this study, a hybrid recommendation system has been proposed that will increase the efficiency of the systems in pharmacies. A new system which called LightGCN+ aims to improve the [LightGCN](https://github.com/kuandeng/LightGCN) by adding item-item relations next to user-item relations. In addition three datasets have been proposed about drug/preparation.

## Environment Requirement
The code has been tested running under Python 3.6.5. The required packages are as follows:
* tensorflow == 1.11.0
* numpy == 1.14.3
* scipy == 1.1.0
* sklearn == 0.19.1
* cython == 0.29.15

## Examples to run a 3-layer LightGCN
The instruction of commands has been clearly stated in the codes (see the parser function in LightGCN/utility/parser.py).

### Drug-Relation-90
* Command
```
python LightGCN.py --dataset Drug-Relation-90 --regs [1e-4] --embed_size 64 --layer_size [64,64,64] --lr 0.001 --batch_size 2048 --epoch 1000
```

* Example Output log (Not Real):
```
eval_score_matrix_foldout with cpp
n_users=12765, n_items=8415
n_interactions=33217
n_train=23000, n_test=10217, sparsity=0.00031
      ...
Epoch 1 [20.3s]: train==[0.46925=0.46911 + 0.00014]
Epoch 2 [25.1s]: train==[0.21866=0.21817 + 0.00048]
      ...
Epoch 879 [81.6s + 31.3s]: test==[0.13271=0.12645 + 0.00626 + 0.00000], recall=[0.18201], precision=[0.05601], ndcg=[0.15555]
Early stopping is trigger at step: 5 log:0.18201370537281036
Best Iter=[38]@[32829.6]	recall=[0.40890], precision=[0.02151], ndcg=[0.20539]
```

### Drug-Relation-180
* Command
```
python LightGCN.py --dataset Drug-Relation-180 --regs [1e-4] --embed_size 64 --layer_size [64,64,64] --lr 0.001 --batch_size 2048 --epoch 1000
```

* Example Output log (Not Real):
```
eval_score_matrix_foldout with cpp
n_users=19210, n_items=9793
n_interactions=57763
n_train=27000, n_test=23763, sparsity=0.00031
      ...
Epoch 1 [20.3s]: train==[0.46925=0.46911 + 0.00014]
Epoch 2 [25.1s]: train==[0.21866=0.21817 + 0.00048]
      ...
Epoch 879 [81.6s + 31.3s]: test==[0.13271=0.12645 + 0.00626 + 0.00000], recall=[0.18201], precision=[0.05601], ndcg=[0.15555]
Early stopping is trigger at step: 5 log:0.18201370537281036
Best Iter=[38]@[32829.6]    recall=[0.48135], precision=[0.02727], ndcg=[0.21539]
```

### Drug-Relation-270
* Command
```
python LightGCN.py --dataset Drug-Relation-270 --regs [1e-4] --embed_size 64 --layer_size [64,64,64] --lr 0.001 --batch_size 2048 --epoch 1000
```

* Example Output log (Not Real):
```
eval_score_matrix_foldout with cpp
n_users=22701, n_items=10282
n_interactions=75480
n_train=40000, n_test=35480, sparsity=0.00032
      ...
Epoch 1 [20.3s]: train==[0.46925=0.46911 + 0.00014]
Epoch 2 [25.1s]: train==[0.21866=0.21817 + 0.00048]
      ...
Epoch 879 [81.6s + 31.3s]: test==[0.13271=0.12645 + 0.00626 + 0.00000], recall=[0.18201], precision=[0.05601], ndcg=[0.15555]
Early stopping is trigger at step: 5 log:0.18201370537281036
Best Iter=[38]@[32829.6]    recall=[0.51632], precision=[0.02945], ndcg=[0.22551]
```

NOTE : the duration of training and testing depends on the running environment.

## Dataset
We provide three processed datasets: Drug-Relation-90, Drug-Relation-180 and Drug-Relation-270.

* `train.txt`
  * Train file.
  * Each line is a user with her/his positive interactions with items: userID\t a list of itemID\n.

* `test.txt`
  * Test file (positive instances).
  * Each line is a user with her/his positive interactions with items: userID\t a list of itemID\n.
  * Note that here we treat all unobserved interactions as the negative instances when reporting performance.

* `item_item.txt`
  * Item-Item relation file.
  * Each line is a item with its positive interactions with other items: itemID\t a list of itemID\n.
  
* `user_list.txt`
  * User file.
  * Each line is a triplet (org_id, remap_id) for one user, where org_id and remap_id represent the ID of the user in the original and our datasets, respectively.
  
* `item_list.txt`
  * Item file.
  * Each line is a triplet (org_id, remap_id) for one item, where org_id and remap_id represent the ID of the item in the original and our datasets, respectively.
