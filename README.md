# Decoupled Graph Convolution Network for Inferring Substitutable and Complementary Items (CIKM 2020 Applied Research Track)

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
| Name                  | Type            | Description   |
| :-------------        |:-------------   |:------------- |
| mode                  | enum(str)            | train, evaluate or save_embedding. |
| data_dir              | str             | directory of the specified dataset (e.g., ./euler_data/Beauty). |
| max_id                | int             | maximum node id, i.e., the number of nodes - 1. |
| sparse_feature_max_id | list(int)       | list of maximum feature id. | 
| dim                   | int             | dimensionality of hidden layers. |
| embedding_dim         | int             | dimensionality of feature embeddings. |
| num_negs              | int             | number of negative samples during training. |
| fanouts               | list(int)       | numbers of sampled neighbors. |
| model                 | str             | model to be trained (e.g., DecGCN). |
| model_dir             | str             | directory to save/load a model. |
| batch_size            | int             | training batch size. |
| optimizer             | enum(str)            | training optimizer (e.g., adam or sgd). |
| learning_rate         | float           | learning rate for training. |
| num_epochs            | int             | number of passes over the training data. |
| log_steps             | int             | number of batches to print the log info. |

## Citation

Please kindly cite the paper if this repo is helpful :)

```
@inproceedings{liu2020decoupled,
  title={Decoupled Graph Convolution Network for Inferring Substitutable and Complementary Items},
  author={Liu, Yiding and Gu, Yulong and Ding, Zhuoye and Gao, Junchao and Guo, Ziyi and Bao, Yongjun and Yan, Weipeng},
  booktitle={CIKM'20},
  year={2020}
}
```
