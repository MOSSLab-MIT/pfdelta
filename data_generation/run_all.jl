using Distributed

using Pkg
Pkg.activate(@__DIR__)

using OPFLearn
using Debugger
using Dates
using Printf
using JSON

# Define the cases
timestamp = Dates.format(Dates.now(), "yyyy-mm-dd_HH-MM-SS")
base_save_path = joinpath("my_results", timestamp)
mkpath(base_save_path)

cases = [
        (
        K = 1000,
        name = "case2000",
        file = "pglib_opf_case2000_goc.m",
        net_path = "pglib",
        save_path = joinpath(base_save_path, "case2000/none"),
        perturb_topology_method = "none"
    )
    # (
    #     K = 56_000,
    #     name = "case57",
    #     file = "pglib_opf_case57_ieee.m",
    #     net_path = "pglib",
    #     save_path = joinpath(base_save_path, "case57/none"),
    #     perturb_topology_method = "none"
    # ),
    # (
    #     K = 29_000,
    #     name = "case57",
    #     file = "pglib_opf_case57_ieee.m",
    #     net_path = "pglib",
    #     save_path = joinpath(base_save_path, "case57/n-1"),
    #     perturb_topology_method = "n-1"
    # ), 
    # (
    #     K = 20_000,
    #     name = "case57",
    #     file = "pglib_opf_case57_ieee.m",
    #     net_path = "pglib",
    #     save_path = joinpath(base_save_path, "case57/n-2"),
    #     perturb_topology_method = "n-2"
    # ),
    # (
    #     K = 56_000,
    #     name = "case14",
    #     file = "pglib_opf_case14_ieee.m",
    #     net_path = "pglib",
    #     save_path = joinpath(base_save_path, "case14/none"),
    #     perturb_topology_method = "none"
    # ),
    # (
    #     K = 29_000,
    #     name = "case14",
    #     file = "pglib_opf_case14_ieee.m",
    #     net_path = "pglib",
    #     save_path = joinpath(base_save_path, "case14/n-1"),
    #     perturb_topology_method = "n-1"
    # ), 
    # (
    #     K = 20_000,
    #     name = "case14",
    #     file = "pglib_opf_case14_ieee.m",
    #     net_path = "pglib",
    #     save_path = joinpath(base_save_path, "case14/n-2"),
    #     perturb_topology_method = "n-2"
    # )
]

# Storage for results
results = []

# Run each case
for case in cases
    println("Running $(case.name)...")
    mkpath(case.save_path)
    start_time = time()
    success = true
    projection_feasible_counter = missing

    try
        results_opflearn, projection_feasible_counter = create_samples(case.file, case.K;
            net_path    = case.net_path,
            save_path   = case.save_path,
            save_while  = true,
            print_level = 1, 
            perturb_topology_method = case.perturb_topology_method, 
            perturb_costs_method = "shuffle"
        )
    catch e
        println("❌ Error in $(case.name): $e")
        success = false
    end

    elapsed = time() - start_time
    push!(results, (case.name, success, round(elapsed, digits=2), case.K))

    summary_data = [
        Dict(
            "case" => case.name,
            "success" => success,
            "runtime_sec" => round(elapsed, digits=2),
            "number_of_samples" => case.K,
            "timestamp" => string(Dates.now()),
            "samples_from_projection" => projection_feasible_counter
        ) 
    ]
    
    open(joinpath(case.save_path, "summary.json"), "w") do io
        JSON.print(io, summary_data)
    end

    println("📄 Saved summary to $(case.save_path)/summary.json")
end
