# Decoupled Graph Convolution Network for Inferring Substitutable and Complementary Items (CIKM 2020)

## How To Use

### Preprocessing
- Step1: Download meta data from https://nijianmo.github.io/amazon/index.html.
- Step2: Put the meta data file in <tt>./preprocessing/raw_data/</tt>.
- Step3: Set the dataset name (i.e., <tt>$dataset</tt>) in run.sh, run preprocessing by <tt>cd preprocessing; sh run.sh</tt>.

The compressed data files (i.e., .dat files) will be put in <tt>./euler_data/$dataset_name</tt>.


### Training 

#### Example of training on Amazon Beauty dataset:
```python
python run_loop.py --data_dir=./euler_data/Beauty --max_id=114792 --sparse_feature_max_id=11,45,11179 \
                   --dim=128 --embedding_dim=16 --num_negs=5 --fanouts=5,5 \
                   --model=DecGCN --model_dir=ckpt \
                   --batch_size=512 --optimizer=adam --learning_rate=1e-4 --num_epochs=20 --log_steps=20
```

#### Parameters:
- data_dir:
- max_id: 
- sparse_feature_max_id
- dim
- embedding_dim
- num_negs
- fanouts
- model
- model_dir
- batch_size
- optimizer
- learning_rate
- num_epochs
- log_steps
