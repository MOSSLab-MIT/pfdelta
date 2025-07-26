using Pkg
# Activate the local environment
Pkg.activate(@__DIR__)

# Instantiate only if Manifest.toml is missing
if !isfile(joinpath(@__DIR__, "Manifest.toml"))
    Pkg.instantiate()
end

import PowerModels as PM
import Random
using MATLAB
import JSON
import Glob
using Debugger
using FilePathsBase, FileIO
PM.silence()

include("create_cpf_samples.jl")

# Setup MATLAB call  
ENV["MATLAB_HOME"] = "/Applications/MATLAB_R2024b.app" # add your MATLAB path here (only intel MATLAB works for mac)

# Argument parsing
if length(ARGS) < 2
    println("Usage: julia main_close2inf.jl <case_name> <perturbation> <want_all>")
    println("Example: julia main_close2inf.jl case14 n-1 true")
    exit(1)
end

case_name = ARGS[1]
topology_perturb = ARGS[2]
want_all = length(ARGS) >= 3 ? ARGS[3] : false

# Define number of training samples
n_samples_nose_train = 1800
n_samples_around_nose_train = 4 

# Define number of test samples
n_samples_nose_test = 200

solved_cases_path = joinpath("/Volumes/Crucial X9", case_name, topology_perturb)

# Create training data
split = "train"
create_close2infeasible(solved_cases_path, n_samples_nose_train, n_samples_around_nose_train, split; save_all=want_all)

# Create testing data
split = "test"
create_close2infeasible(solved_cases_path, n_samples_nose_test, nothing, split)

# Make sure you have the right number of files after it is done
files = Glob.glob("sample_*.json", joinpath(solved_cases_path, "close2inf_train", "around_nose"))
println("Number of files in close2inf_train_around_nose: ", length(files))

files = Glob.glob("sample_*.json", joinpath(solved_cases_path, "close2inf_train", "nose"))
println("Number of files in close2inf_train_nose: ", length(files))

files = Glob.glob("sample_*.json", joinpath(solved_cases_path, "close2inf_test", "nose"))
println("Number of files in close2inf_test: ", length(files))