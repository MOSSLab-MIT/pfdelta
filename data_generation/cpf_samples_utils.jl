const N_SAMPLES_NOSE_TRAIN = 1800 # TODO: double-check
const N_SAMPLES_AROUND_NOSE_TRAIN = 4  # TODO: double-check
const N_SAMPLES_NOSE_TEST = 200 # TODO: double-check
matlab_scripts_dir = abspath(joinpath(@__DIR__, "matlab"))
mat"addpath($matlab_scripts_dir)"

function create_close2infeasible(data_dir, case_name, topology_perturb; delete_int_files=false, run_analysis_mode=false)

    # TODO: add docstrings!
    # TODO: solved_cases path needs to be more descriptive of the fact that this is the data dir where all case data is stored.

    # Resolve the path to the solved cases
    solved_cases_path = joinpath(data_dir, case_name, topology_perturb)
    
    selected_cases_idx_train, selected_cases_idx_test, selected_cases_idx_analysis = parse_shuffle_file(data_dir, topology_perturb)
    
    # Set up dirs to save files
    dirs = create_dirs(solved_cases_path)

    # Create samples
    selected_cases_idx_train !== nothing && create_train_samples(selected_cases_idx_train, dirs["train"]; delete_int_files=delete_int_files)

    selected_cases_idx_test !== nothing && create_test_samples(selected_cases_idx_test, dirs["test"]; delete_int_files=delete_int_files)

    if run_analysis_mode
        selected_cases_idx_analysis !== nothing && create_train_samples(selected_cases_idx_analysis, dirs["analysis"]; delete_int_files=delete_int_files, n_nose_samples=100)
    end
    # TODO: implement sample validation
end

# Helper functions
function parse_shuffle_file(data_dir, topology_perturb)
    raw_shuffle_path = joinpath(data_dir, "shuffle_files", topology_perturb, "raw_shuffle.json") # TODO: this may change as we modify the folder structure
    if isfile(raw_shuffle_path)
        shuffled_idx = JSON.parsefile(raw_shuffle_path) # TODO: not sure this is a good name for this given the strucuture of the json file.
        sorted_keys = sort(parse.(Int, collect(keys(shuffled_idx)))) # TODO: why are you even doing this?
        selected_cases_idx_train = [shuffled_idx[string(k)] for k in sorted_keys[1:end-2000]] # TODO: do not hard code the 2000 here
        selected_cases_idx_test = [shuffled_idx[string(k)] for k in sorted_keys[end-2000+1:end]] # TODO: do not hard code the 2000 here
        return (selected_cases_idx_train, selected_cases_idx_test, nothing)
    else
        num_samples_debug = 10 # TODO: make this a parameter
        @info "raw_shuffle.json not found, considering 2000 samples for analysis mode."
        return nothing, nothing, collect(1:2000)
    end
end

function create_dirs(solved_cases_path)
    # TODO: use either the word dir or path consistently

    paths = Dict{String, NamedTuple}()
    for split in ["train", "test", "analysis"]
        close2inf_path   = joinpath(solved_cases_path, "close2inf_" * split) 
        mpc_save_path    = joinpath(close2inf_path, "generated_mpcs")
        raw_hard_save    = joinpath(close2inf_path, "raw")
        nose_dir         = joinpath(close2inf_path, "nose")

        # always created
        mkpath.((
            close2inf_path,
            mpc_save_path,
            raw_hard_save,
            nose_dir,
            joinpath(raw_hard_save, "non_converging"), # had this for debugging originally, may remove
        ))

        dirs_to_clean = [mpc_save_path, raw_hard_save, nose_dir]

        # only for train split
        if split in ["train", "analysis"]
            around_nose_dir = joinpath(close2inf_path, "around_nose")
            mkpath(around_nose_dir)
            push!(dirs_to_clean, around_nose_dir)
        else
            around_nose_dir = nothing
        end

        # Warn before deleting
        for d in dirs_to_clean
            if !isempty(readdir(d))
                @warn "Deleting existing contents in folder" folder=d
            end
        end

        # Empty them
        empty_folder.(dirs_to_clean)

        # Update paths dict
        paths[split] = (
            base=close2inf_path,
            mpcs=mpc_save_path,
            raw=raw_hard_save,
            nose=nose_dir,
            around_nose=around_nose_dir,
            solved_cases=solved_cases_path,
        )
    end
    return paths
end

function create_train_samples(selected_cases_idx_train, train_dirs; delete_int_files=false, n_nose_samples=N_SAMPLES_NOSE_TRAIN)

    # Get paths to save samples
    close2inf_path = train_dirs.base
    mpc_save_path = train_dirs.mpcs
    raw_hard_save_path = train_dirs.raw
    nose_dir = train_dirs.nose
    around_nose_dir = train_dirs.around_nose
    solved_cases_path = train_dirs.solved_cases

    # Initialize counters
    successful_files = 0
    i = 0
    p = Progress(n_nose_samples; desc="Creating train samples", dt=0.5)

    while successful_files < n_nose_samples

        i += 1 # shuffle file keys are 1-indexed
        current_sample_idx = selected_cases_idx_train[i]

        # Convert current sample from PowerModels to MATPOWER format
        current_net_path  = create_matpower_file(solved_cases_path, mpc_save_path, current_sample_idx)

        # Solve CPF using MATPOWER
        sample_cpf_save_path = joinpath(raw_hard_save_path, "cpf_sample_$(current_sample_idx)")
        mkpath(sample_cpf_save_path)
        mat"cpf_success = solve_cpf($current_net_path, $sample_cpf_save_path);"
        @mget cpf_success

        if cpf_success
            # Validate CPF samples for current sample
            skip_sample = validate_cpf_samples(sample_cpf_save_path)
            
            if skip_sample
                # Delete the directory associated with this sample
                @warn "Skipping sample $current_sample_idx due to invalid CPF samples, deleting associated files."
                rm(sample_cpf_save_path; force=true, recursive=true)
            else                
                # Save nose sample
                save_nose_samples(sample_cpf_save_path, nose_dir, current_sample_idx)

                # Create file tuples for around-the-nose samples
                file_tuples = create_file_tuples(sample_cpf_save_path)

                # Save around-the-nose samples
                save_around_nose_samples(file_tuples, around_nose_dir, current_sample_idx) # consider creating subdirs here for each sample?

                successful_files += 1
                next!(p; showvalues = [
                    (:done, successful_files),
                    (:total, n_nose_samples),
                    (:pct, round(100 * successful_files / n_nose_samples; digits=1))
                ])

                # Optional: Delete directory with cpf files
                if delete_int_files
                    rm(sample_cpf_save_path; force=true, recursive=true)
                end
            end
        end
    end
    finish!(p)
end

function create_test_samples(selected_cases_idx_test, test_dirs; delete_int_files=false)

    # Get paths to save samples
    close2inf_path = test_dirs.base
    mpc_save_path = test_dirs.mpcs
    raw_hard_save_path = test_dirs.raw
    nose_dir = test_dirs.nose
    solved_cases_path = test_dirs.solved_cases

    # Initialize counters
    successful_files = 0
    i = 0
    p = Progress(N_SAMPLES_NOSE_TEST; desc="Creating test samples", dt=5)


    while successful_files < N_SAMPLES_NOSE_TEST

        i += 1 # shuffle file keys are 1-indexed 
        current_sample_idx = selected_cases_idx_test[i]

        # Convert current sample from PowerModels to MATPOWER format
        current_net_path  = create_matpower_file(solved_cases_path, mpc_save_path, current_sample_idx)

        # Solve CPF using create_matpower_file
        sample_cpf_save_path = joinpath(raw_hard_save_path, "cpf_sample_$(current_sample_idx)")
        mkpath(sample_cpf_save_path)
        mat"cpf_success = solve_cpf($current_net_path, $sample_cpf_save_path);"
        cpf_success = @mget cpf_success

        if cpf_success
            # Validate CPF samples for current sample
            skip_sample = validate_cpf_samples(sample_cpf_save_path)
            
            if skip_sample
                # Delete the directory associated with this sample
                @warn "Skipping sample $current_sample_idx due to invalid CPF samples, deleting associated files."
                rm(sample_cpf_save_path; force=true, recursive=true)
            else                
                # Save nose sample only 
                save_nose_samples(sample_cpf_save_path, nose_dir, current_sample_idx)

                successful_files += 1
                next!(p)

                # Optional: Delete directory with cpf files
                if delete_int_files
                    rm(sample_cpf_save_path; force=true, recursive=true)
                end
            end
        end
        update!(p; desc="Creating test samples ($(successful_files)/$(N_SAMPLES_NOSE_TEST))")
    end
end

function validate_cpf_samples(sample_cpf_save_path)::Bool
    files = readdir(sample_cpf_save_path; join=true)
    valid_count = 0

    for file in files
        base = basename(file)
        endswith(base, "_nose.m") && continue

        m = match(r"^sample_(\d+)_lam_(m?\d+p\d+)\.m$", base)
        m === nothing && continue

        lam_str = replace(m.captures[2], "p" => ".", "m" => "-")
        lam = tryparse(Float64, lam_str)
        lam === nothing && continue

        if lam <= 0
            return true   # skip sample
        else 
            valid_count += 1
        end
    end

    if valid_count < N_SAMPLES_AROUND_NOSE_TRAIN
        return true # skip sample, not enough around the nose points
    end

    return false  # safe to keep
end

function save_nose_samples(sample_cpf_save_path, nose_dir, current_sample_idx)
    nose_file = Glob.glob("sample_$(current_sample_idx)_nose.m", sample_cpf_save_path)[1]
    solved_net = PM.parse_file(nose_file)
    json_path = joinpath(nose_dir, "sample_$(current_sample_idx)_nose.json") 
    json_dict_nose = Dict("lambda" => nothing, "solved_net" => solved_net) # TODO: would be nice to add lambda value here corresponding to the nose.
    open(json_path, "w") do io
        write(io, JSON.json(json_dict_nose))
    end
end 

function save_around_nose_samples(file_tuples, around_nose_dir, current_sample_idx)
        sorted_files = sort(file_tuples, by = x -> abs(x[2]), rev=true)
        selected_files = sorted_files[2:N_SAMPLES_AROUND_NOSE_TRAIN+1] 
        for (file, lam) in selected_files
            base = basename(file)
            solved_net = PM.parse_file(file)
            json_dict = Dict("lambda" => lam, "solved_net" => solved_net)
            filepath = joinpath(around_nose_dir, replace(base, ".m" => ".json")) # TODO: not defined
            if isfile(filepath)
                @warn "Overwriting existing file: $filepath" 
            end
            open(filepath, "w") do io
                write(io, JSON.json(json_dict))
            end
        end
end

function create_file_tuples(sample_cpf_save_path)
    files = readdir(sample_cpf_save_path; join=true)
    file_tuples = []

    for file in files
        base = basename(file)
        endswith(base, "_nose.m") && continue

        m = match(r"^sample_(\d+)_lam_(m?\d+p\d+)\.m$", base)
        m === nothing && continue

        lam_str = replace(m.captures[2], "p" => ".", "m" => "-")
        lam = tryparse(Float64, lam_str)
        lam === nothing && continue

        push!(file_tuples, (file, lam))
    end

    return file_tuples
end

function validate_samples()
    files_nose = Glob.glob("sample_*.json", joinpath(close2inf_path, "nose")) # TODO: not defined
    @assert length(files_nose) == n_samples_nose_train "Got $(length(files_nose)) instead of $(n_samples_nose_train)" # TODO: not defined
    
    if split === "train" # TODO: not defined
        files_around = Glob.glob("sample_*.json", joinpath(close2inf_path, "around_nose")) # TODO: not defined
        @assert length(files_around) == n_samples_nose_train * n_samples_around_nose_train "Got $(length(files_around)) instead of $(n_samples_nose_train * n_samples_around_nose_train)" # TODO: not defined
    end
end

# Old helper functions

function create_matpower_file(solved_cases_path, mpc_save_path, idx)
    json_file = "sample_$(idx).json"
    sample = JSON.parsefile(joinpath(solved_cases_path,"raw", json_file))
    net = sample["network"]
    solution = sample["solution"]["solution"]
    PM.update_data!(net, solution)
    update_vg!(net, solution)
    fix_branch_flows!(net)
    PM.export_matpower(joinpath(mpc_save_path, "sample_$(idx).m"), net)
    return joinpath(mpc_save_path, "sample_$(idx).m")
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

function get_condition_num_and_NR(results, raw_path, sample_idx, pv_curve_data)
    pattern = "sample_$(sample_idx)_lam_*.m"
    matches = Glob.glob(pattern, raw_path)

    if isempty(matches)
        @warn "No file found for sample $sample_idx"
        return
    end


    for filename in matches
        m = match(r"lam_(\d+p\d+)", filename)
        if m === nothing
            @warn "Filename $filename did not match lambda pattern"
            continue
        end

        lam = parse(Float64, replace(m.captures[1], "p" => "."))
        net = PM.parse_file(filename)
        pv_curve_data[sample_idx][lam] = deepcopy(net)
        net = PM.make_basic_network(net)
        J = PM.calc_basic_jacobian_matrix(net)

        pf_solution = PM.compute_ac_pf(net; flat_start=true)
        converged = pf_solution["termination_status"] == true
        iterations = pf_solution["iterations"]
        solve_time = pf_solution["solve_time"]

        results[sample_idx][lam] = (cond(Array(J), 2), converged, iterations, solve_time)
    end
    
end

