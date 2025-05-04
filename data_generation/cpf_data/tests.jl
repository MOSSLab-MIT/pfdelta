using Pkg
Pkg.activate(".")
import PowerModels as PM
import Random
import MATLAB
import JSON
import Glob
using Debugger
using FilePathsBase, FileIO

ENV["MATLAB_HOME"] = "/Applications/MATLAB_R2024b.app" # intel MATLAB
mat"disp('MATLAB is now talking to Julia!')"

include("create_close2infeasible.jl")

solved_cases_path = joinpath("..","my_results/2025-04-28_11-28-43/case14_n")
k = 2.5
n_nose = 2
n_around_nose = 4

create_close2infeasible(solved_cases_path, k, n_nose, n_around_nose)