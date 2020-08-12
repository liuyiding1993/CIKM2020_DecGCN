# Decoupled Graph Convolution Network for Inferring Substitutable and Complementary Items (CIKM 2020)

## How To Use

### Preprocessing
- Step1: Download meta data from https://nijianmo.github.io/amazon/index.html.
- Step2: Put the meta data file in <tt>./preprocessing/raw_data/</tt>.
- Step3: Set the dataset name (i.e., <tt>$dataset</tt>) in run.sh, run preprocessing by <tt>cd preprocessing; sh run.sh</tt>.

The compressed data files (i.e., .dat files) will be put in <tt>./euler_data/$dataset_name/</tt>.


### Training 

#### Example of training on Amazon Beauty dataset:
```python
python run_loop.py --mode=train --data_dir=./euler_data/Beauty \
                   --max_id=114791 --sparse_feature_max_id=10,44,11178 \
                   --dim=128 --embedding_dim=16 --num_negs=5 --fanouts=5,5 \
                   --model=DecGCN --model_dir=ckpt --batch_size=512 \
                   --optimizer=adam --learning_rate=1e-4 --num_epochs=20 --log_steps=20
```

#### Parameters:
| Name                  | Type            | Description  |
| :-------------         |:-------------:  | -----:|
| data_dir              | str             | $1600 |
| max_id                | int             |   $12 |
| sparse_feature_max_id | list(int)       |    $1 |

-  (str): directory of the specified dataset (e.g., ./euler_data/Beauty).
-  (int): maximum node id, i.e., the number of nodes - 1.
-  (\[int,...\]): list of maximum feature id.  
- dim (int): dimensionality of hidden layers.
- embedding_dim (int): dimensionality of feature embeddings.
- num_negs (int): number of negative samples during training.
- fanouts (\[int,...\]): numbers of sampled neighbors.
- model (str): model to be trained (e.g., DecGCN).
- model_dir (str): directory to save/load a model. 
- batch_size: training batch size.
- optimizer: training optimizer (e.g., adam or sgd).
- learning_rate: learning rate for training.
- num_epochs: number of passes over the training data.
- log_steps: number of batches to print the log info.
