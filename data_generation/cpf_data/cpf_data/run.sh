#!/bin/bash

# Define your list of case names and perturbations
case_names=("case30") #"case57" "case118") # "case500" "case2000")
perturbations=("n-2")

# Loop over all combinations
for case_name in "${case_names[@]}"; do
  for perturbation in "${perturbations[@]}"; do
    echo "Running: julia main_close2inf.jl $case_name $perturbation"
    julia main_close2inf.jl "$case_name" "$perturbation"
  done
done
