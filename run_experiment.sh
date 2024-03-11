#!/bin/bash
# Script to run set of experiments
for d in airline_passenger_satisfaction credit_card stellar
do
    for m in cartography confidence
    do
        # for e in 15
        # do
        echo "Setup: dataset - $d, ranking_mode: $m, relaxed - False"
        python -m src.models.train_model -d $d -m $m
        echo "Setup: dataset - $d, ranking_mode: $m, relaxed - True"
        python -m src.models.train_model -d $d -r True -m $m
        # done
    done
done