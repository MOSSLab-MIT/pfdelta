using Pkg
Pkg.activate("data_generation/")
import JSON3
import JSON
import PowerModels as PM
using Statistics
using Printf
using DataFrames
using CSV
using Logging

const DATA_DIR = "data"
const WARM_UP_RUNS = 5
const OUT_DIR = "runtimes_results"
const TOPO_PERTURB = ["n", "n-1", "n-2"]
const LOG_FILE = ""

include("runtimes_utils.jl")

function init_logger(case_name)
    mkpath(joinpath(OUT_DIR, "logs"))
    logfile = open(joinpath(OUT_DIR, "logs", "runtime_$(case_name).log"), "w")
    global_logger(SimpleLogger(logfile, Logging.Info))
end

function main()
    case_name = ARGS[1]
    near_infeasible_flag = parse(Bool, ARGS[2]) 

    init_logger(case_name)

    for topology_perturb in TOPO_PERTURB
        @info "Processing case: $case_name with topology perturbation: $topology_perturb"
        if !near_infeasible_flag
            test_samples_idx = parse_shuffle_file(DATA_DIR, topology_perturb; case_2000_flag=(case_name=="case2000"))
        else
            test_samples_idx = nothing
        end
        test_networks = create_test_networks(test_samples_idx, case_name, topology_perturb; near_infeasible_flag=near_infeasible_flag)

        @info "Running warm-up iterations..."
        run_warm_up(test_networks, WARM_UP_RUNS)

        @info "Starting runtime measurements..."
        get_runtimes(test_networks, case_name, topology_perturb, near_infeasible_flag ? "nose" : "raw")
    end

    @info "Analyzing runtimes..."
    analyze_runtimes(case_name; topologies=["n", "n-1", "n-2"])
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end