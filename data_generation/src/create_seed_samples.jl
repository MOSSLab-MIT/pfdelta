function create_seed_samples(; net, radius, num_samples,
                             path_to_save, starting_id,
                             seed_id=-1, perturb_topology_method, perturb_costs_method,
                             print_level=0, opf_solver=JuMP.optimizer_with_attributes(Ipopt.Optimizer, "tol" => TOL))

    k = 0
    i = 0

    while k < num_samples
        i += 1
        net_perturbed = deepcopy(net)

        perturb_load!(net_perturbed, radius)

        perturb_topology!(net_perturbed; method=perturb_topology_method)

        perturb_costs!(net_perturbed; method=perturb_costs_method)

        result, feasible, results_pfdelta = run_ac_opf_pfdelta(net_perturbed, print_level=print_level, solver=opf_solver)

        if print_level > 0
            println("OPF SUCCESS: ", feasible)
        end

        if feasible
            k += 1
            store_feasible_sample_json(k + starting_id, net_perturbed, results_pfdelta, path_to_save; seed_id=seed_id)

            if print_level > 0
                println("Samples: $(k) / $(num_samples),\t Iterations: $(i)")
            end
        end
    end
end
