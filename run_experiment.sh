#!/bin/bash
# Script to run set of experiments
# Gammas to test: 2 2.5 5 -> to be continued credit_card fashion_mnist
# Datasets to test:
# DONE: airline_passenger_satisfaction stellar
for n in {1..50}
do
    echo "ITERATION: $n"
    for d in airline_passenger_satisfaction stellar credit_card fashion_mnist
    do
        # echo "Setup: dataset - $d, ranking_mode: None, relaxed - False, ranked - False"
        # python -m src.train_model -d $d -e 8 -g 1.5 
        for m in cartography confidence
        do
            # echo "Setup: dataset - $d, ranking_mode: $m, relaxed - False, ranked - True"
            # python -m src.train_model -d $d -m $m -e 8 -g 1.5 -a 0.1 -bt 2.5 -rnk
            for g in 1.5 2 2.5 5
            do
                echo "Setup: dataset - $d, ranking_mode: $m, relaxed - True, ranked - False, gamma - $g"
                python -m src.train_model -d $d -m $m -e 8 -a 0.1 -bt 2.5 -g $g -r
                echo "Setup: dataset - $d, ranking_mode: $m, relaxed - True, ranked - True, gamma - $g"
                python -m src.train_model -d $d -m $m -e 8 -a 0.1 -bt 2.5 -g $g -rnk -r
            done 
            # done
        done
    done
done