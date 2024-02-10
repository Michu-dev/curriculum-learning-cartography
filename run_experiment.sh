#!/bin/bash
# Script to run set of experiments
for d in airline_passenger_satisfaction credit_card stellar
do
    for b in 10 100 1000
    do
        for e in 2 4 8
        do
            echo "Setup: dataset - $d, batch_size - $b, epochs - $e, relaxed - False"
            python -m src.models.train_model -d $d -b $b -e $e
            echo "Setup: dataset - $d, batch_size - $b, epochs - $e, relaxed - True"
            python -m src.models.train_model -d $d -r True -b $b -e $e
        done
    done
done