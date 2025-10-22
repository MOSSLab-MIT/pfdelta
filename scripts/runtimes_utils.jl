function parse_shuffle_file(data_dir, topology_perturb; case_2000_flag=false)
    if case_2000_flag
        raw_shuffle_path = joinpath(DATA_DIR, "shuffle_files", topology_perturb, "raw_shuffle_case2000.json")
    else
        raw_shuffle_path = joinpath(DATA_DIR, "shuffle_files", topology_perturb, "raw_shuffle.json") # TODO: this may change as we modify the folder structure
    end
    if isfile(raw_shuffle_path)
        shuffled_idx = JSON.parsefile(raw_shuffle_path) # TODO: not sure this is a good name for this given the strucuture of the json file.
        sorted_keys = sort(parse.(Int, collect(keys(shuffled_idx)))) # TODO: why are you even doing this?
        selected_cases_idx_test = [shuffled_idx[string(k)] + 1 for k in sorted_keys[end-2000+1:end]]
        return selected_cases_idx_test
    else
        @error("Shuffle file not found at $raw_shuffle_path")
    end
end

function run_warm_up(test_networks, warm_up_runs)
    for sample_idx in Iterators.take(sort(collect(keys(test_networks))), warm_up_runs)
        network = test_networks[sample_idx]
        try
            PM.compute_ac_pf(network; flat_start=true)
        catch e
            @warn "Warm-up failed on sample $sample_idx: $e"
        end
    end
end

function create_test_networks(test_samples_idx, case_name, topological_perturb; near_infeasible_flag=false)
    test_networks = Dict{Int, Any}()

    if !near_infeasible_flag
        for idx in test_samples_idx
            fname = "sample_$(idx).json"
            fpath = joinpath(DATA_DIR, case_name, topological_perturb, "raw", fname)
            if isfile(fpath)
                sample_data = JSON.parsefile(fpath)

                # Fix vg and branch flows (unclear if needed but it can't hurt). Note how this is only needed for feasible samples.
                update_vg!(sample_data["network"], sample_data["solution"]["solution"])
                fix_branch_flows!(sample_data["network"])

                # Update test networks dictionary
                test_networks[idx] = sample_data["network"]
            else
                @warn "File not found: $fpath"
            end
        end
    elseif near_infeasible_flag
        nose_cases_path = joinpath(DATA_DIR, case_name, topological_perturb, "nose", "test")
        for fname in readdir(nose_cases_path)
            if endswith(fname, ".json")
                m = match(r"sample_(\d+)_nose\.json", fname)
                if m !== nothing
                    sample_idx = parse(Int, m.captures[1])
                    fpath = joinpath(nose_cases_path, fname)
                    sample_data = JSON.parsefile(fpath)
                    test_networks[sample_idx] = sample_data["solved_net"]
                end
            end
        end
    end
    return test_networks
end

function get_runtimes(test_networks, case_name, topology_perturb, sample_type)
    for run in 1:3
        run_dir = joinpath(OUT_DIR, case_name, topology_perturb, sample_type, "run_$run")
        mkpath(run_dir)
        current_run_data = Vector{Tuple{Int, Float64, Bool}}() # (sample_idx, solve_time, converged)

        for sample_idx in sort(collect(keys(test_networks)))
            fname = "sample_$(sample_idx).json"
            network = test_networks[sample_idx]

            try
                pf_solution = PM.compute_ac_pf(network; flat_start=true)
                converged = pf_solution["termination_status"] == true
                push!(current_run_data, (Int(sample_idx), pf_solution["solve_time"], converged))

                # Save power flow solution if converged
                if converged
                    solved_net = deepcopy(network)
                    PM.update_data!(solved_net, pf_solution["solution"])
                    open(joinpath(run_dir, "solved_sample_$(sample_idx).json"), "w") do io
                        JSON3.write(io, solved_net; indent=4)
                    end
                end
                    
            catch e
                @warn "Error on sample index $(sample_idx): $e, changing bus type now." # Note that this should never happen now
                        
                # Fix bus type
                network_retry = deepcopy(network)
                change_bus_type!(network_retry)

                try
                    pf_solution = PM.compute_ac_pf(network_retry; flat_start=true)
                    converged = pf_solution["termination_status"] == true

                    # Save power flow solution if converged
                    if converged
                        solved_net = deepcopy(network_retry)
                        PM.update_data!(solved_net, pf_solution["solution"])
                        open(joinpath(run_dir, "solved_$(fname)"), "w") do io
                            JSON3.write(io, solved_net; indent=4)
                        end
                    end
                    push!(current_run_data, (Int(sample_idx), pf_solution["solve_time"], converged))
                    @info "Bus type change worked for current sample"
                catch e2
                    @warn "Retry also failed on current sample: $e2"
                    push!(current_run_data, (Int(sample_idx), 0.0, false))
                end
            end
        end

        runtime_data = [Dict("sample_idx" => i, "solve_time" => t, "converged" => c) for (i, t, c) in current_run_data]
        open(joinpath(run_dir, "runtime_NR_test.json"), "w") do io
            JSON3.write(io, runtime_data; indent=4)  
        end
    end
end

function analyze_runtimes(case_name, near_infeasible_flag::Bool;
                          topologies=["none","n-1","n-2"],
                          runs=1:3,
                          data_dir=DATA_DIR)

    # Prepare csv output paths
    results_filename = "runtimes_$(case_name)_$(near_infeasible_flag ? "nose" : "raw").csv"
    summary_filename = "runtimes_summary_$(case_name)_$(near_infeasible_flag ? "nose" : "raw").csv"
    results_out_csv = joinpath(OUT_DIR, results_filename)
    summary_out_csv = joinpath(OUT_DIR, summary_filename)

    # Initialize DataFrames
    results_df = DataFrame(run=Int[], sample_idx=Int[], topology_perturb=String[], solve_time=Float64[], converged=Bool[])
    summary_df = DataFrame(
        run = Int[],
        run_mean = Float64[],
        pct_converged = Float64[],
    )

    # Create resuls DataFrame
    for topology_perturb in topologies
        sample_type = near_infeasible_flag ? "nose" : "raw"

        for run in runs
            run_dir = joinpath(OUT_DIR, case_name, topology_perturb, sample_type, "run_$run")
            runtime_file = joinpath(run_dir, "runtime_NR_test.json")

            if isfile(runtime_file)
                runtime_data = JSON.parsefile(runtime_file)

                for entry in runtime_data
                    push!(results_df, (
                        run,
                        entry["sample_idx"],
                        topology_perturb,
                        entry["solve_time"],
                        entry["converged"]
                    ))
                end
            else
                @warn "Runtime file not found: $runtime_file"
            end
        end
    end

    # Create summary DataFrame (considering all topologies together)
    for run in runs
        filtered_df = filter(row -> row.run == run, results_df)
        run_mean = mean(filter(row -> row.converged, filtered_df).solve_time)
        pct_converged = 100 * sum(filtered_df.converged) / nrow(filtered_df)

        push!(summary_df, (
            run,
            run_mean,
            pct_converged
        ))
    end

    # Print summary of mean of means, std, and overall convergence rate
    overall_mean = mean(summary_df.run_mean)
    overall_std = std(summary_df.run_mean)
    overall_pct_converged = 100 * sum(results_df.converged) / nrow(results_df)
    @info "Overall Mean Solve Time: $(overall_mean) seconds"
    @info "Overall Std Dev of Solve Time: $(overall_std) seconds"
    @info "Overall Percentage of Converged Cases: $(overall_pct_converged)%"

    # Save DataFrames to CSV
    CSV.write(results_out_csv, results_df)
    CSV.write(summary_out_csv, summary_df)
    @info "Results saved to $results_out_csv and $summary_out_csv"
end

function change_bus_type!(net)
    PV_buses = [bus_key for (bus_key, bus) in net["bus"] if bus["bus_type"] == 2]
    pv_bus_set = Set(parse(Int, k) for k in PV_buses)

    PV_bus_to_gen_map = Dict{String, Vector{Dict{String, Any}}}() # stores mapping from bus_idx to gens at that bus

    # Populate PV_bus_to_gen_map
    for (_, gen) in net["gen"]
        gen_bus = gen["gen_bus"]
        if gen_bus in pv_bus_set
            push!(get!(PV_bus_to_gen_map, string(gen_bus), Vector{Dict{String,Any}}()), gen)
        end
    end

    # Check if all gens are out for a given PV bus and change bus type
    for bus_key in PV_buses
        gens = PV_bus_to_gen_map[bus_key]
        if all(gen["gen_status"] == 0 for gen in gens)
            net["bus"][bus_key]["bus_type"] = 1
            @warn "Bus $bus_key was PV and is now a PQ bus (no active gens)"
        end
    end
end

# TODO: do you need this?
function update_vg!(net, solution)
    gen_data = net["gen"]
    bus_solution = solution["bus"]

    for (gen_id, gen) in gen_data
        if gen["gen_status"] == 1
            bus_id = string(gen["gen_bus"])
            if haskey(bus_solution, bus_id)
                gen["vg"] = bus_solution[bus_id]["vm"]
            else
                @warn "No bus voltage solution found for gen_bus = $bus_id"
            end
        end
    end
end

# TODO: do you need this?
function fix_branch_flows!(net)
    branch_data = net["branch"]
    for (branch_id, branch) in branch_data
        if branch["br_status"] == 0
            for key in ["pt", "pf", "qt", "qf"]
                if !haskey(branch, key)
                     branch[key] = 0
                end
            end
        end
    end
end
