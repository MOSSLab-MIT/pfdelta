# Environment Setup
using Pkg
Pkg.activate(@__DIR__)
Pkg.instantiate() # TODO: do we always need this? Maybe add some check to Manifest?
 
# Imports
import PowerModels as PM
import Random
using MATLAB
import JSON
import TOML
import Glob
using FilePathsBase, FileIO
PM.silence()

include("create_cpf_samples.jl")

function main()
    if length(ARGS) < 1
        println("Usage: julia main_close2inf.jl <config.toml>")
        exit(1)
    end

    cfg_path = ARGS[1]
    cfg = TOML.parsefile(cfg_path)

    # required keys
    for k in ("data_dir", "case_name", "topology_perturb")
        haskey(cfg, k) || error("Missing key '$k' in $cfg_path")
    end

    # set MATLAB env var if present
    if haskey(cfg, "matlab") && haskey(cfg["matlab"], "home")
        ENV["MATLAB_ROOT"] = cfg["matlab"]["home"]
    end

    create_close2infeasible(cfg["data_dir"], cfg["case_name"], cfg["topology_perturb"])
end

main()