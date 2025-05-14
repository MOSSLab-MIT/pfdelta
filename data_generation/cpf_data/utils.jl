function flat_start_NR!(net)
    for (_,bus) in net["bus"]
        bus["va"] = 0.0
        if bus["bus_type"] == 1
            bus["vm"] = 1.0
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

        flat_start_NR!(net)
        pf_solution = PM.compute_ac_pf(net; flat_start=true)
        converged = pf_solution["termination_status"] == true

        results[sample_idx][lam] = (cond(Array(J), 2), converged)
    end
    
end

