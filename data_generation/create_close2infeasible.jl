# Environment Setup
using Pkg
Pkg.activate(@__DIR__)
Pkg.instantiate() # TODO: do we always need this? Maybe add some check to Manifest?
 
# Imports
import PowerModels as PM
import Random
import JSON
import TOML
import Glob
using FilePathsBase # TODO: is this used? 
using ProgressMeter
PM.silence()

include("cpf_samples_utils.jl")

function main()
    if length(ARGS) < 1
        println("Usage: julia main_close2inf.jl <config.toml>") # TODO: need to clarify paths here
        exit(1)
    end

    cfg_path = ARGS[1]
    cfg = TOML.parsefile(cfg_path)

    # required keys
    for k in ("data_dir", "case_name", "topology_perturb")
        haskey(cfg, k) || error("Missing key '$k' in $cfg_path")
    end

    # set MATLAB
    ensure_matlab_root!(cfg)
    using MATLAB # don't like having the import here

    create_close2infeasible(
        cfg["data_dir"], cfg["case_name"], cfg["topology_perturb"];
        delete_int_files=cfg["delete_intermediate"], 
        run_analysis_mode=cfg["analysis_mode"]
    )
end

function ensure_matlab_root!(cfg::Dict)
    # If config provides a path, prefer it (cluster or local)
    if haskey(cfg, "matlab") && haskey(cfg["matlab"], "home")
        home = String(cfg["matlab"]["home"])
        if !isempty(home)
            # Accept either .../R20xxa or .../R20xxa/bin; strip /bin if present
            home = endswith(home, "/bin") ? dirname(home) : home
            ENV["MATLAB_ROOT"] = home
            return
        end
    end
    # Otherwise, if module exposed `matlab` on PATH, infer MATLAB_ROOT
    matlab_cmd = Sys.which("matlab")
    if matlab_cmd !== nothing
        # e.g., /orcd/software/matlab/R2024a/bin/matlab -> /orcd/software/matlab/R2024a
        ENV["MATLAB_ROOT"] = dirname(dirname(matlab_cmd))
        return
    end
    # Last resort: leave unset and let MATLAB.jl try; but emit a helpful message
    @warn "MATLAB_ROOT not set and 'matlab' not on PATH. On clusters, load a module (e.g., `module load matlab/matlab-2024a`) or set [matlab].home in the config."
end

main()