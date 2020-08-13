#!/bin/bash

dataset=Beauty

mkdir ../euler_data
mkdir ../euler_data/$dataset

mkdir stats 
mkdir data 
mkdir tmp


echo "---------------- step 1: feature filter ----------------"
python 1_feature_filter.py $dataset
echo "--------------------------------------------------------"

echo "---------------- step 2: edge extraction ---------------"
python 2_edge_extractor.py $dataset
echo "--------------------------------------------------------"

echo "---------------- step 3: edge filter -------------------"
python 3_edge_filter.py $dataset
echo "--------------------------------------------------------"

echo "---------------- step 4: data formulation --------------"
python 4_data_formulator.py $dataset
echo "--------------------------------------------------------"

echo "---------------- step 5: compress to binary ------------"
python -m euler.tools -c ./config/meta.json -i ./data/$dataset.json -o ../euler_data/$dataset/$dataset.dat
echo "--------------------------------------------------------"

