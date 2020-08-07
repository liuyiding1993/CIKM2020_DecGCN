#!/bin/bash

dataset=Beauty

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

echo "---------------- step 5: convert to euler --------------"
python3 5_convert_to_euler.py $dataset
echo "--------------------------------------------------------"
