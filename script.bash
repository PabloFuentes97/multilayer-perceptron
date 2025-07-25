#!/bin/bash

data="data.cvs"
split_data="evaluation.py"
train="clean_train.py"
train_data="data_training.csv"
test="clean_test.py"
test_data="data_test.csv"

for i in {1..5}
    do 
        echo "Model $i: "
        python3 $split_data $data
        python3 $train $train_data
        python3 $test $test_data
done

