function create_close2infeasible(solved_cases_path, n_nose, n_around_nose, split; save_all=false)
    
    raw_shuffle_path = joinpath(solved_cases_path, "raw_shuffle.json")
    if isfile(raw_shuffle_path)
        shuffled_idx = JSON.parsefile(raw_shuffle_path)
        sorted_keys = sort(parse.(Int, collect(keys(shuffled_idx))))

        if split == "train"
            selected_cases_idx = [shuffled_idx[string(k)] for k in sorted_keys[1:n_nose+10]] # the 10 is to have some 10 extra files in case we don't get enough around the nose points
        elseif split == "test"
            selected_cases_idx = [shuffled_idx[string(k)] for k in sorted_keys[end-n_nose-10+1:end]]
        end
    else
        println("raw_shuffle.json not found, using all samples in debug mode.")
        split = "debug"
    
        raw_dir = joinpath(solved_cases_path, "raw")
        json_files = Glob.glob("*.json", raw_dir)
        selected_cases_idx = collect(1:length(json_files))
    end

    close2inf_path = joinpath(solved_cases_path, "close2inf_" * split) 
    mpc_save_path =  joinpath(close2inf_path, "generated_mpcs")
    mkpath(close2inf_path)
    mkpath(mpc_save_path)

    raw_hard_save_path = joinpath(close2inf_path, "raw")
    mkpath(raw_hard_save_path)

    # to avoid more files during re runs
    empty_folder(mpc_save_path) 
    empty_folder(raw_hard_save_path)
    empty_folder(joinpath(close2inf_path, "nose"))
    empty_folder(joinpath(close2inf_path, "around_nose"))

    create_matpower_files(solved_cases_path, mpc_save_path, selected_cases_idx)
    
    mat"""
    generate_close2inf($mpc_save_path, $raw_hard_save_path);
    """

    split_files(close2inf_path, n_nose, n_around_nose, split; save_all=save_all)

    files_nose = Glob.glob("sample_*.json", joinpath(close2inf_path, "nose"))
    @assert length(files_nose) == n_nose "Got $(length(files_nose)) instead of $(n_nose)"
    
    if split === "train"
        files_around = Glob.glob("sample_*.json", joinpath(close2inf_path, "around_nose"))
        @assert length(files_around) == n_nose * n_around_nose "Got $(length(files_around)) instead of $(n_nose * n_around_nose)"
    end

end

function create_matpower_files(solved_cases_path, save_path, selected_cases_idx)
    for idx in selected_cases_idx
        json_file = "sample_$(idx).json"
        sample = JSON.parsefile(joinpath(solved_cases_path,"raw", json_file))
        net = sample["network"]
        solution = sample["solution"]["solution"]
        PM.update_data!(net, solution)
        update_vg!(net, solution)
        fix_branch_flows!(net)
        PM.export_matpower(joinpath(save_path, "sample_$(idx).m"), net)
    end
end

function split_files(close2inf_path, n_nose, n_around_nose, split; save_all=false)
    nose_dir = joinpath(close2inf_path, "nose")
    around_nose_dir = joinpath(close2inf_path, "around_nose")
    mkpath(nose_dir)
    mkpath(around_nose_dir)

    files = Glob.glob("sample_*.m", joinpath(close2inf_path, "raw"))
    
    file_dict = Dict{String, Vector{Tuple{String, Float64}}}()

    for file in files
        base = basename(file)
        if endswith(base, "_nose.m")
            continue
        else
            m = match(r"sample_(\d+)_lam_(\d+p\d+)\.m", base)
            if m !== nothing
                idx = m.captures[1]
                lam = parse(Float64, replace(m.captures[2], "p" => "."))
                push!(get!(file_dict, idx, Tuple{String, Float64}[]), (file, lam))
            end
        end
    end


    filtered_dict = Dict{String, Vector{Tuple{String, Float64}}}()
    if split == "train"
        for (idx, file_tuples) in file_dict
            if save_all || (length(file_tuples) - 1 >= n_around_nose)
                filtered_dict[idx] = file_tuples
            else
                @warn "Skipping index $(idx): not enough around-the-nose samples (have $(length(file_tuples) - 1), need $(n_around_nose))"
            end
        end
    elseif split == "test"
        filtered_dict = file_dict
    end

    selected_idxs = sort(collect(keys(filtered_dict))) 
    if length(selected_idxs) < n_nose
        error("Only $(length(selected_idxs)) valid nose points available, but requested $n_nose.")
    end
    selected_idxs = selected_idxs[1:n_nose]

    # For each nose file (based on index), get around-the-nose samples
    for idx in selected_idxs
        if split == "train"
            file_tuples = filtered_dict[idx]
            sorted = sort(file_tuples, by = x -> abs(x[2]), rev=true)
            selected = save_all ? sorted : Iterators.take(Iterators.drop(sorted, 1), n_around_nose)

            for (file, lam) in selected
                base = basename(file)
                solved_net = PM.parse_file(file)
                json_dict = Dict("lambda" => lam, "solved_net" => solved_net)
                filepath = joinpath(around_nose_dir, replace(base, ".m" => ".json"))
                open(filepath, "w") do io
                    write(io, JSON.json(json_dict))
                end
            end
        end

        # Also save the corresponding nose file now
        nose_file = joinpath(close2inf_path, "raw", "sample_$(idx)_nose.m")
        if isfile(nose_file)
            solved_net = PM.parse_file(nose_file)
            json_path = joinpath(nose_dir, "sample_$(idx)_nose.json")
            json_dict = Dict("lambda" => nothing, "solved_net" => solved_net)
            open(json_path, "w") do io
                write(io, JSON.json(json_dict))
            end
        else
            @warn "Nose file not found for index $idx. Skipping."
        end
    end
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