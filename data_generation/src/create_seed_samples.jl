function create_seed_samples(; net, radius, num_samples, base_case,
    path_to_save, starting_id, seed_id=-1, perturb_topology_method, perturb_costs_method,
    print_level=0, opf_solver=JuMP.optimizer_with_attributes(Ipopt.Optimizer, "tol" => TOL))

    k = 0
    i = 0
    # Copy modified load onto base case to use for new perturbations
    base_case = deepcopy(base_case)
    num_loads = length(base_case["load"])
    for i in 1:num_loads
        base_case["load"][string(i)]["pd"] = net["load"][string(i)]["pd"]
        base_case["load"][string(i)]["qd"] = net["load"][string(i)]["qd"]
    end
    # Now expand each seed
    while k < num_samples
        i += 1
        net_perturbed = deepcopy(base_case)

        perturb_load!(net_perturbed, radius)

        perturb_topology!(net_perturbed; method=perturb_topology_method)

        perturb_costs!(net_perturbed; method=perturb_costs_method)

        result, feasible, results_pfdelta = run_ac_opf_pfdelta(net_perturbed, print_level=print_level, solver=opf_solver)

        if print_level > 1
            println("OPF SUCCESS: ", feasible)
        end

        if feasible
            k += 1
            store_feasible_sample_json(k + starting_id, net_perturbed, results_pfdelta, path_to_save; seed_id=seed_id)
            println("Seed: $(seed_id),\t Samples: $(k) / $(num_samples),\t Iterations: $(i)")
        end
    end
end
