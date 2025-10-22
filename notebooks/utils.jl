function get_condition_num_and_NR(results, raw_path, sample_idx, pv_curve_data)
    # Folder structure: raw_path/cpf_sample_<idx>/
    sample_dir = joinpath(raw_path, "cpf_sample_$(sample_idx)")
    if !isdir(sample_dir)
        @warn "Sample directory not found" sample_dir sample_idx
        return
    end

    # Regex: lam_<digits>(p<digits>)?.m or .json  (e.g., lam_1p25.m → 1.25)
    lam_re = r"lam_(\d+(?:p\d+)?)\.(m|json)$"

    processed = 0
    for path in readdir(sample_dir; join=true)
        fname = basename(path)
        m = match(lam_re, fname)
        m === nothing && continue

        # Parse lambda value (e.g., "1p25" → 1.25)
        lam_str = replace(m.captures[1], "p" => ".")
        lam = try
            parse(Float64, lam_str)
        catch
            @warn "Failed to parse lambda" fname lam_str
            continue
        end

        try
            # --- Parse and process the network ---
            net = PM.parse_file(path)
            pv_curve_data[sample_idx][lam] = deepcopy(net)

            net_basic = PM.make_basic_network(net)
            J = PM.calc_basic_jacobian_matrix(net_basic)

            pf_solution = PM.compute_ac_pf(net_basic; flat_start=true)
            converged = pf_solution["termination_status"] == true
            solve_time = pf_solution["solve_time"]

            # Condition number of the Jacobian
            κ = cond(Array(J), 2)
            results[sample_idx][lam] = (κ, converged, solve_time)

            processed += 1

        catch err
            @warn "Failed on file" fname error=err
            continue
        end
    end

    if processed == 0
        @warn "No CPF files processed for sample" sample_idx sample_dir
    else
        @info "Processed $processed CPF files for sample $sample_idx"
    end

    return
end