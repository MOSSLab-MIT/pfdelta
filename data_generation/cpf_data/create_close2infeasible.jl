function create_close2infeasible(solved_cases_path, k, 
                                n_nose, n_around_nose)

    shuffled_idx = JSON.parsefile(joinpath(solved_cases_path, "raw_shuffle.json"))
    sorted_keys = sort(parse.(Int, collect(keys(shuffled_idx))))

    train_test_idx = Dict(
        "train" => [shuffled_idx[string(k)] for k in sorted_keys[1:1800]],
        "test"  => [shuffled_idx[string(k)] for k in sorted_keys[end-199:end]]
    )
    
    @assert length(train_test_idx["train"]) == 1800 "Expected 1800 values in first_values"
    @assert length(train_test_idx["test"]) == 200 "Expected 200 values in last_values"

    for split in keys(train_test_idx)
        
        close2inf_path = joinpath(solved_cases_path, "close2inf_" * split) 
        mpc_save_path =  joinpath(close2inf_path, "generated_mpcs")
        mkpath(close2inf_path)
        mkpath(mpc_save_path)

        raw_hard_save_path = joinpath(close2inf_path, "raw")
        mkpath(raw_hard_save_path)

        empty_folder(mpc_save_path)
        empty_folder(raw_hard_save_path)
        empty_folder(joinpath(close2inf_path, "around_nose"))
        empty_folder(joinpath(close2inf_path, "nose"))

        create_matpower_files(solved_cases_path, mpc_save_path, train_test_idx[split])

        mat"""
        generate_close2inf($mpc_save_path, $k, $raw_hard_save_path);
        """

        split_files(close2inf_path, n_around_nose)
    end
end

function create_matpower_files(solved_cases_path, save_path, selected_cases_idx)
    for idx in selected_cases_idx
        json_file = "sample_$(idx).json"
        sample = JSON.parsefile(joinpath(solved_cases_path,"raw", json_file))
        net = sample["network"]
        solution = sample["solution"]["solution"]
        PM.update_data!(net, solution)
        PM.export_matpower(joinpath(save_path, "sample_$(idx).m"), net)
    end
end

function split_files(close2inf_path, n_around_nose)
    nose_dir = joinpath(close2inf_path, "nose")
    around_nose_dir = joinpath(close2inf_path, "around_nose")
    mkpath(nose_dir)
    mkpath(around_nose_dir)

    files = Glob.glob("sample_*.m", joinpath(close2inf_path, "raw"))
    
    file_dict = Dict{String, Vector{Tuple{String, Float64}}}()

    for file in files
        base = basename(file)
        if endswith(base, "_nose.m")
            target_path = joinpath(nose_dir, base)
            cp(file, target_path, force=true)
            solved_net = PM.parse_file(target_path)
            filepath = joinpath((nose_dir, replace(base, ".m" => ".json")))
            open(filepath, "w") do io
                write(io, JSON.json(solved_net))
            end
        else
            m = match(r"sample_(\d+)_lam_(\d+p\d+)\.m", base)
            if m !== nothing
                idx = m.captures[1]
                lam = parse(Float64, replace(m.captures[2], "p" => "."))
                push!(get!(file_dict, idx, []), (file, lam))
            end
        end
    end

    for (_, file_lams) in file_dict
        sorted = sort(file_lams; by=x -> x[2], rev=true)
        selected = sorted[2:n_around_nose + 1]
        for (file, _) in selected
            base = basename(file)
            target_path = joinpath(around_nose_dir, base)
            cp(file, target_path; force=true)
            solved_net = PM.parse_file(target_path)
            filepath = joinpath((around_nose_dir, replace(base, ".m" => ".json")))
            open(filepath, "w") do io
                write(io, JSON.json(solved_net))
            end
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