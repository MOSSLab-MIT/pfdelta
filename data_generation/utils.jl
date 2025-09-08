# helper functions for data checks

function get_condition_num_NR(net)
    J = PM.calc_basic_jacobian_matrix(net)
    cond_num = cond(Array(J), 2)
    return cond_num
end

function get_NR_data(net)
    pf_solution = PM.compute_ac_pf(net; flat_start=true)
    converged = pf_solution["termination_status"] in SOLVED_STATUSES
    solve_time = pf_solution["solve_time"]
    return pf_solution, converged, solve_time
end

function solve_warm_PF(net)
    # Use the current state of `net` as a warm start
    pf_solution = PM.compute_ac_pf(net)
    return pf_solution
end

function get_solution_diff(net, pf_solution, converged)
    if converged
        diffs = Float64[]

        # --- bus voltages ---
        for (bus_id, bus) in net["bus"]
            vm_net = bus["vm"]
            va_net = bus["va"]
            vm_sol = pf_solution["solution"]["bus"][bus_id]["vm"]
            va_sol = pf_solution["solution"]["bus"][bus_id]["va"]

            push!(diffs, abs(vm_net - vm_sol))
            push!(diffs, abs(va_net - va_sol))
        end

        # --- generator powers ---
        for (gen_id, gen) in net["gen"]
            pg_net = gen["pg"]
            qg_net = gen["qg"]

            pg_sol = pf_solution["solution"]["gen"][gen_id]["pg"]
            qg_sol = pf_solution["solution"]["gen"][gen_id]["qg"]

            push!(diffs, abs(pg_net - pg_sol))
            push!(diffs, abs(qg_net - qg_sol))
        end

        return maximum(diffs)
    else 
        return NaN
    end
end

## all plotting functions 
function plot_condition_numbers(sample_results)
    sort!(sample_results, by = x -> x[1])
    lambdas = [x[1] for x in sample_results]
    cond_nums = [x[2] for x in sample_results]
    return bar(lambdas, cond_nums,
        xlabel = "λ",
        ylabel = "Condition Number",
        title  = "Condition Numbers by λ",
        legend = false)
end

function plot_NR_convergence(sample_results)
    sort!(sample_results, by = x -> x[1])
    lambdas = [x[1] for x in sample_results]
    converged = [x[3] for x in sample_results]
    return bar(lambdas, converged,
        xlabel = "λ",
        ylabel = "Converged (1=True, 0=False)",
        title  = "NR Convergence by λ",
        legend = false)
end

function plot_NR_solve_time(sample_results)
    sort!(sample_results, by = x -> x[1])
    lambdas = [x[1] for x in sample_results]
    solve_times = [x[4] for x in sample_results]
    return bar(lambdas, solve_times,
        xlabel = "λ",
        ylabel = "Solve Time (s)",
        title  = "NR Solve Time by λ",
        legend = false)
end

function plot_solution_difference(sample_results)
    sort!(sample_results, by = x -> x[1])
    lambdas = [x[1] for x in sample_results]
    solution_diffs = [x[5] for x in sample_results]
    return bar(lambdas, solution_diffs,
        xlabel = "λ",
        ylabel = "Solution Difference",
        title  = "Solution Difference by λ",
        legend = false)
end

function plot_PV_curves(sample_results, buses)
    plots = Plot[]

        for bus in buses
            # collect (λ, vm) pairs
            lam_vm_pairs = Tuple{Float64,Float64}[]
            for tup in sample_results
                lam = tup[1]
                net = tup[end]             # assume `net` is the last element of the tuple
                vm  = net["bus"][bus]["vm"]
                push!(lam_vm_pairs, (lam, vm))
            end

            # sort by λ and unzip
            sort!(lam_vm_pairs, by = x -> x[1])
            lams = getindex.(lam_vm_pairs, 1)
            vms  = getindex.(lam_vm_pairs, 2)

            # indices for highlights
            n = length(lams)
            start10   = max(n - 10 + 1, 1)           # for optional last-10 window
            idx_last  = n
            idx_green = n >= 2 ? (max(n - 4, 1):(n - 1)) : (1:0)  # empty if n < 2

            # full PV curve with highlighted last points
            p = plot(lams, vms;
                    label = "",
                    xlabel = "Continuation parameter (λ)",
                    ylabel = "Voltage Magnitude (VM) at bus $bus",
                    linecolor = :black,
                    marker = :circle,
                    legend = false,
                    dpi = 300)

            if !isempty(idx_green)
                scatter!(p, lams[idx_green], vms[idx_green];
                        markercolor = :green, marker = :circle, label = "")
            end

            if n >= 1
                scatter!(p, [lams[idx_last]], [vms[idx_last]];
                        markercolor = :red, marker = :circle, label = "")
            end

            push!(plots, p)
        end

        # 2x2 grid (assumes `buses` has 4 elements)
        return plot(plots...; layout = (2, 2), size = (1200, 900))
end




