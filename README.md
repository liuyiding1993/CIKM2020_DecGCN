# Decoupled Graph Convolution Network for InferringSubstitutable and Complementary Items (CIKM 2020)

## How To Use

### Preprocessing
- Step1: Download meta data from https://nijianmo.github.io/amazon/index.html.
- Step2: Put the meta data file in ./preprocessing/raw_data/.
- Step3: Set the dataset name in run.sh, run preprocessing by sh ./preprocessing/run.sh.

The compressed data files (i.e., .dat files) will be put in <tt>./euler_data/$dataset_name</tt>.


### Training 
