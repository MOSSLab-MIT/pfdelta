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
                line_to_remove = rand(branch_keys)
                net["branch"][line_to_remove]["br_status"] = 0
            else
                # Remove a non-slack generator
                gen_to_remove = rand(gen_keys)
                while net["gen"][gen_to_remove]["index"] in slack_bus_gen_idx
                    gen_to_remove = rand(gen_keys)
                end
                net["gen"][gen_to_remove]["gen_status"] = 0
            end

        elseif method == "N-2"
            if random_num < 1/3
                # Remove two random lines
                selected_lines = shuffle(branch_keys)[1:2]
                for key in selected_lines
                    net["branch"][key]["br_status"] = 0
                end
            elseif random_num > 1/3
                # Remove two random generators (non-slack only)
                selected_gens = shuffle(gen_keys)[1:2]
                gen_indices = [net["gen"][g]["index"] for g in selected_gens]

                while any(i -> i in slack_bus_gen_idx, gen_indices)
                    selected_gens = shuffle(gen_keys)[1:2]
                    gen_indices = [net["gen"][g]["index"] for g in selected_gens]
                end

                for g in selected_gens
                    net["gen"][g]["gen_status"] = 0
                end
            else
                # Remove one line and one non-slack generator
                line_to_remove = rand(branch_keys)
                net["branch"][line_to_remove]["br_status"] = 0

                gen_to_remove = rand(gen_keys)
                while net["gen"][gen_to_remove]["index"] in slack_bus_gen_idx
                    gen_to_remove = rand(gen_keys)
                end
                net["gen"][gen_to_remove]["gen_status"] = 0
            end

        else
            error("Invalid method specified: $method. Choose from 'none', 'N-1', or 'N-2'.")
        end

        # Only return if the network is still connected
        is_connected_adj_matrix(net) && return
    end
end

# benchmark this (could be taking longer than solving the opf)
function is_connected_adj_matrix(net)
    branches = net["branch"]
    buses = collect(keys(net["bus"]))
    bus_index_map = Dict(b => i for (i, b) in enumerate(buses))  # bus_i → 1-based index for Graphs.jl

    g = SimpleGraph(length(buses))

    for branch in values(branches)
        if branch["br_status"] == 1
            f = branch["f_bus"]
            t = branch["t_bus"]
            add_edge!(g, bus_index_map[f], bus_index_map[t])
        end
    end

    return is_connected(g)
end

function perturb_costs!(net; method="none")
    method = uppercase(method)

    if method == "SHUFFLE"
        gen_keys = collect(keys(case["gen"]))
        shuffled_keys = shuffle(gen_keys)
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