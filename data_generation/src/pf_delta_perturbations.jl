using Graphs: SimpleGraph, add_edge!, is_connected

function perturb_topology!(net; method="none") 
    branch_keys = collect(keys(net["branch"]))
    gen_keys = collect(keys(net["gen"]))

    slack_bus_idx = [bus["bus_i"] for bus in values(net["bus"]) if bus["bus_type"] == 3]
    slack_bus_gen_idx = [gen["index"] for gen in values(net["gen"]) if gen["gen_bus"] in slack_bus_idx]

    method = uppercase(method)

    if method == "NONE"
        return
    end

    # Save original statuses (do not overwrite lines/gens that were off in the base case)
    original_br_status = Dict(k => net["branch"][k]["br_status"] for k in branch_keys)
    original_gen_status = Dict(k => net["gen"][k]["gen_status"] for k in gen_keys)

    # Only pick from active lines and generators
    active_branch_keys = [k for k in branch_keys if original_br_status[k] == 1]
    active_gen_keys = [k for k in gen_keys if original_gen_status[k] == 1]

    while true
        # Reset branches and gen status
        for k in branch_keys
            net["branch"][k]["br_status"] = original_br_status[k]
        end

        for k in gen_keys
            net["gen"][k]["gen_status"] = original_gen_status[k]
        end

        random_num = rand()

        if method == "N-1"
            if random_num < 0.5
                # Remove a random line
                line_to_remove = rand(active_branch_keys)
                net["branch"][line_to_remove]["br_status"] = 0
            else
                # Remove a non-slack generator
                gen_to_remove = rand(active_gen_keys)
                while net["gen"][gen_to_remove]["index"] in slack_bus_gen_idx
                    gen_to_remove = rand(active_gen_keys)
                end
                net["gen"][gen_to_remove]["gen_status"] = 0
            end

        elseif method == "N-2"
            if random_num < 1/3
                # Remove two random lines
                selected_lines = Random.shuffle(active_branch_keys)[1:2]
                for key in selected_lines
                    net["branch"][key]["br_status"] = 0
                end
            elseif random_num > 2/3
                # Remove two random generators (non-slack only)
                selected_gens = Random.shuffle(active_gen_keys)[1:2]
                gen_indices = [net["gen"][g]["index"] for g in selected_gens]

                while any(i -> i in slack_bus_gen_idx, gen_indices)
                    selected_gens = Random.shuffle(active_gen_keys)[1:2]
                    gen_indices = [net["gen"][g]["index"] for g in selected_gens]
                end

                for g in selected_gens
                    net["gen"][g]["gen_status"] = 0
                end
            else
                # Remove one line and one non-slack generator
                line_to_remove = rand(active_branch_keys)
                net["branch"][line_to_remove]["br_status"] = 0

                gen_to_remove = rand(active_gen_keys)
                while net["gen"][gen_to_remove]["index"] in slack_bus_gen_idx
                    gen_to_remove = rand(active_gen_keys)
                end
                net["gen"][gen_to_remove]["gen_status"] = 0
            end

        else
            error("Invalid method specified: $method. Choose from 'none', 'N-1', or 'N-2'.")
        end
        
        # Only return if the network is still connected
        if is_graph_connected(net)
            return
        end
    end
end

function is_graph_connected(net)
    nb = length(net["bus"])
    g = SimpleGraph(nb)

    for (_, branch) in net["branch"]
        if branch["br_status"] == 1
            f = branch["f_bus"]
            t = branch["t_bus"]
            add_edge!(g, f, t)
        end
    end

    return is_connected(g)
end

function perturb_costs!(net; method="none")
    method = uppercase(method)

    if method == "SHUFFLE"
        gen_keys = collect(keys(net["gen"]))
        shuffled_keys = Random.shuffle(gen_keys)
        temp_case = deepcopy(net)
        for (orig_key, shuffled_key) in zip(gen_keys, shuffled_keys)
            net["gen"][orig_key]["cost"] = temp_case["gen"][shuffled_key]["cost"]
        end

    elseif method == "NONE"
        return 
    else
        error("Invalid method specified: $method. Choose from 'none', 'shuffle'.")
    end
end

function perturb_load!(net, r)
    # Step 1: Create active power demand vector
    load_ids = sort(collect(keys(net["load"])))
    pd0 = [net["load"][id]["pd"] for id in load_ids]

    n_loads = length(pd0)

    success = false
    pd_new = similar(pd0)
    while !success
        # Sample uniformly from a ball around pd0
        pd_new = sample_ball(pd0, r)

        if all(pd_new .>= 0.0)
            success = true
        end
    end

    # Sample random power factors
    pf_range = (0.8, 1.0)  # can adjust
    sampled_pf = (pf_range[2] - pf_range[1]) * rand(n_loads) .+ pf_range[1]

    # Step 3: Calculate new reactive powers
    qd_new = pd_new .* tan.(acos.(sampled_pf))

    # Step 4: Update network loads
    for (i, id) in enumerate(load_ids)
        net["load"][id]["pd"] = pd_new[i]
        net["load"][id]["qd"] = qd_new[i]
    end

    return nothing
end

function sample_ball(center::Vector{Float64}, r::Float64)
    n = length(center)

    # Step 1: Random direction
    z = randn(n)
    z = z / norm(z)

    # Step 2: Random radius with correct scaling
    u = rand()
    radius = r * u^(1/n)

    # Step 3: Build sample
    sample = center .+ radius .* z
    return sample
end
