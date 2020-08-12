# Decoupled Graph Convolution Network for InferringSubstitutable and Complementary Items (CIKM 2020)

## How To Use

### Preprocessing
- Step1: Download meta data from https://nijianmo.github.io/amazon/index.html.
- Step2: Put the meta data file in <tt>./preprocessing/raw_data/</tt>.
- Step3: Set the dataset name (i.e., <tt>$dataset</tt>) in run.sh, run preprocessing by <tt>cd preprocessing; sh run.sh</tt>.

The compressed data files (i.e., .dat files) will be put in <tt>./euler_data/$dataset_name</tt>.


### Training 

Example of training on Amazon Beauty dataset:
```python
python run_loop.py --data_dir=./euler_data/Beauty 
```
