function create_close2infeasible_samples(solved_cases_path, n_nose, n_around_nose, split; save_all=false)
    
    raw_shuffle_path = joinpath(solved_cases_path, "raw_shuffle.json")
    if isfile(raw_shuffle_path)
        shuffled_idx = JSON.parsefile(raw_shuffle_path)
        sorted_keys = sort(parse.(Int, collect(keys(shuffled_idx))))

        if split == "train"
            selected_cases_idx = [shuffled_idx[string(k)] for k in sorted_keys[1:end-2000]]
        elseif split == "test"
            selected_cases_idx = [shuffled_idx[string(k)] for k in sorted_keys[end-2000+1:end]]
        end
    else
        println("raw_shuffle.json not found, using all samples in debug mode.")
        split = "debug"
        raw_dir = joinpath(solved_cases_path, "raw")
        json_files = Glob.glob("*.json", raw_dir)
        selected_cases_idx = collect(1:length(json_files))
    end

    close2inf_path = joinpath(solved_cases_path, "close2inf_" * split) 
    println(close2inf_path)
    mpc_save_path =  joinpath(close2inf_path, "generated_mpcs")
    raw_hard_save_path = joinpath(close2inf_path, "raw")
    nose_dir = joinpath(close2inf_path, "nose")
    around_nose_dir = joinpath(close2inf_path, "around_nose")
    
    mkpath(close2inf_path)
    mkpath(mpc_save_path)
    mkpath(raw_hard_save_path)
    mkpath(nose_dir)
    mkpath(around_nose_dir)

    empty_folder.([mpc_save_path, raw_hard_save_path, nose_dir, around_nose_dir])
    mkpath(joinpath(raw_hard_save_path, "non_converging"))

    i = 0
    successful_files = 0
    around_nose_counter = 0
    while successful_files < n_nose
        i += 1
        current_sample_idx = selected_cases_idx[i]
        create_matpower_file(solved_cases_path, mpc_save_path, current_sample_idx) 
        current_net_path = joinpath(mpc_save_path, "sample_$(current_sample_idx).m")
        mat"cpf_success = solve_cpf($current_net_path, $raw_hard_save_path);"
        cpf_success = @mget cpf_success

        if cpf_success 
            if split == "train"
                current_sample_files = Glob.glob("sample_$(current_sample_idx)_*.m", raw_hard_save_path)
                file_tuples = Tuple{String, Float64}[]
                skip_sample = false
                
                for file in current_sample_files
                    base = basename(file)
                    if !endswith(base, "_nose.m")
                        m = match(r"sample_(\d+)_lam_(m?\d+p\d+)\.m", base)
                        if m !==nothing
                            lam = parse(Float64, replace(m.captures[2], "p" => ".", "m" => "-"))
                            if lam <= 0
                                skip_sample = true
                                break 
                            else
                                push!(file_tuples, (file, lam))                            
                            end
                        end
                    end
                end

                if skip_sample || length(file_tuples) < n_around_nose + 1
                    # delete all files associated with this sample
                    for file in current_sample_files
                        rm(file, force=true)
                    end
                    nose_file = joinpath(raw_hard_save_path, "sample_$(current_sample_idx)_nose.m")
                    rm(nose_file, force=true)
                    continue
                end

                # Save nose
                nose_file = joinpath(close2inf_path, "raw", "sample_$(current_sample_idx)_nose.m")
                solved_net = PM.parse_file(nose_file)
                json_path = joinpath(nose_dir, "sample_$(current_sample_idx)_nose.json")
                json_dict_nose = Dict("lambda" => nothing, "solved_net" => solved_net)
                open(json_path, "w") do io
                    write(io, JSON.json(json_dict_nose))
                    successful_files += 1
                end

                # Save around-the-nose-files
                sorted_files = sort(file_tuples, by = x -> abs(x[2]), rev=true)
                selected_files = sorted_files[2:n_around_nose+1]

                for (file, lam) in selected_files
                    base = basename(file)
                    solved_net = PM.parse_file(file)
                    json_dict = Dict("lambda" => lam, "solved_net" => solved_net)
                    filepath = joinpath(around_nose_dir, replace(base, ".m" => ".json"))
                    if isfile(filepath)
                        @warn "Overwriting existing file: $filepath"
                    end
                    open(filepath, "w") do io
                        write(io, JSON.json(json_dict))
                        around_nose_counter += 1
                    end
                end

            elseif split == "test"
                # Just save around the nose
                nose_file = joinpath(raw_hard_save_path, "sample_$(current_sample_idx)_nose.m")
                solved_net = PM.parse_file(nose_file)
                json_path = joinpath(nose_dir, "sample_$(current_sample_idx)_nose.json")
                json_dict_nose = Dict("lambda" => nothing, "solved_net" => solved_net)
                open(json_path, "w") do io
                    write(io, JSON.json(json_dict_nose))
                    successful_files += 1
                end
            end
        end
        println("Succesful_files: $(successful_files) / $(n_nose)")
    end

    files_nose = Glob.glob("sample_*.json", joinpath(close2inf_path, "nose"))
    @assert length(files_nose) == n_nose "Got $(length(files_nose)) instead of $(n_nose)"
    
    if split === "train"
        files_around = Glob.glob("sample_*.json", joinpath(close2inf_path, "around_nose"))
        @assert length(files_around) == n_nose * n_around_nose "Got $(length(files_around)) instead of $(n_nose * n_around_nose)"
    end
end

function create_matpower_file(solved_cases_path, save_path, idx)
    json_file = "sample_$(idx).json"
    sample = JSON.parsefile(joinpath(solved_cases_path,"raw", json_file))
    net = sample["network"]
    solution = sample["solution"]["solution"]
    PM.update_data!(net, solution)
    update_vg!(net, solution)
    fix_branch_flows!(net)
    PM.export_matpower(joinpath(save_path, "sample_$(idx).m"), net)
end


function empty_folder(path::String)
    isdir(path) || return
    for file in readdir(path; join=true)
        isfile(file) && rm(file; force=true)
        isdir(file) && rm(file; force=true, recursive=true)
    end
end

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