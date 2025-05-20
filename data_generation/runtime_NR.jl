using Pkg
Pkg.activate(".")
import JSON3
import JSON
import PowerModels as PM

function change_bus_type!(network)
    bus_gens = Dict{Int, Vector{Dict{String,Any}}}()
    for (_, gen) in network["gen"]
        bus = gen["gen_bus"]
        push!(get!(bus_gens, bus, Vector{Dict{String,Any}}()), gen)
    end
    
    for (_, bus) in network["bus"]
        bus_idx = bus["bus_i"]

        if !haskey(bus_gens, bus_idx) || all(gen["gen_status"] == 0 for gen in bus_gens[bus_idx])
            bus["bus_type"] = 1 # became a PQ bus
        end
    end
end

case_name = ARGS[1]
topological_perturb = ARGS[2]
hard_case = parse(Bool, ARGS[3])

where_data_lives = "final_data_no_exp"
raw_shuffle_path = joinpath(where_data_lives, case_name, topological_perturb, "raw_shuffle.json")
shuffled_idx = JSON.parsefile(raw_shuffle_path)
sorted_idx_keys = sort(parse.(Int, collect(keys(shuffled_idx))))
test_samples_idx = [shuffled_idx[string(k)] for k in sorted_idx_keys[end-2000+1:end]]


test_networks = Dict{Int, Any}()

if !hard_case
    for idx in test_samples_idx
        fname = "sample_$(idx).json"
        fpath = joinpath(where_data_lives, case_name, topological_perturb, "raw", fname)
        if isfile(fpath)
            sample_data = JSON.parsefile(fpath)
            test_networks[idx] = sample_data["network"]
        else
            @warn "File not found: $fpath"
        end
    end
elseif hard_case
    nose_cases_path = joinpath(where_data_lives, case_name, topological_perturb, "close2inf_test", "nose")
    println(nose_cases_path)
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

type = !hard_case ? "normal" : "nose"

warm_up_runs = 50

for sample_idx in Iterators.take(sort(collect(keys(test_networks))), warm_up_runs)
    network = test_networks[sample_idx]
    try
        PM.compute_ac_pf(network; flat_start=true)
    catch e
        @warn "Warm-up failed on sample $sample_idx: $e"
    end
end

for run in 1:3
    runtime_NR = Vector{Tuple{Float64, Bool}}()

    for sample_idx in sort(collect(keys(test_networks)))
        fname = "sample_$(sample_idx).json"
        network = test_networks[sample_idx]

        try
            pf_solution = PM.compute_ac_pf(network; flat_start=true)
            converged = pf_solution["termination_status"] == true
            push!(runtime_NR, (pf_solution["solve_time"], converged))

            # Save power flow solution if converged
            if converged
                solved_net = deepcopy(network)
                PM.update_data!(solved_net, pf_solution["solution"])
                save_path = joinpath("inference_results_NR_pt2", case_name, topological_perturb, type, "run_$run")
                mkpath(save_path)
                open(joinpath(save_path, "solved_sample_$(sample_idx).json"), "w") do io
                    JSON3.write(io, solved_net; indent=4)
                end
            end
                
        catch e
            @warn "Error on sample index $(sample_idx): $e, changing bus type now."
                    
            # Fix bus type
            change_bus_type!(network)

            try
                pf_solution = PM.compute_ac_pf(network; flat_start=true)
                converged = pf_solution["termination_status"] == true

                # Save power flow solution if converged
                if converged
                    solved_net = deepcopy(network)
                    PM.update_data!(solved_net, pf_solution["solution"])
                    save_path = joinpath("inference_results_NR_pt2", case_name, topological_perturb, type, "run_$run")
                    mkpath(save_path)
                    open(joinpath(save_path, "solved_$fname.json"), "w") do io
                        JSON3.write(io, solved_net; indent=4)
                    end
                end
                push!(runtime_NR, (pf_solution["solve_time"], converged))
                println("Bus type change worked for current sample")
            catch e2
                @warn "Retry also failed on current sample: $e2"
                push!(runtime_NR, (0.0, false))
            end
        end
    end

    runtime_data = [Dict("solve_time" => t, "converged" => conv) for (t, conv) in runtime_NR]
    save_path = joinpath("inference_results_NR_pt2", case_name, topological_perturb, type, "run_$run")
    mkpath(save_path)

    open(joinpath(save_path, "runtime_NR_test.json"), "w") do io
        JSON3.write(io, runtime_data; indent=4)  
    end
end