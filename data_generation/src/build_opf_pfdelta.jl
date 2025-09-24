# TODO: rename this function
function build_opf_power_flow_delta(pm)
    variable_bus_voltage_pfdelta(pm)
    variable_gen_power_pfdelta(pm)
    variable_branch_power_pfdelta(pm)
    PM.variable_dcline_power(pm) # doesn't matter but better to keep it.

    PM.objective_min_fuel_and_flow_cost(pm)

    for i in ids(pm, :ref_buses)
        PM.constraint_theta_ref(pm, i)
    end

    for i in ids(pm, :bus)
        PM.constraint_power_balance(pm, i)
    end

    for i in ids(pm, :branch)
        PM.constraint_ohms_yt_from(pm, i)
        PM.constraint_ohms_yt_to(pm, i)
        PM.constraint_voltage_angle_difference(pm, i) # added back in to see its effects
    end

    for i in ids(pm, :dcline)
        PM.constraint_dcline_power_losses(pm, i)
    end

end


function variable_bus_voltage_pfdelta(pm; nw::Int=PM.nw_id_default)
    # Voltage angle constraints don't change from PowerModels
    PM.variable_bus_voltage_angle(pm)

    # Voltage magnitude 
    vm = var(pm, nw)[:vm] = JuMP.@variable(pm.model,
        [i in ids(pm, nw, :bus)], base_name="$(nw)_vm",
        start = PM.comp_start_value(ref(pm, nw, :bus, i), "vm_start", 1.0)
        )

    # Set general non-negativity
    for i in ids(pm, nw, :bus)
        JuMP.set_lower_bound(vm[i], 0.0)
    end
    
    # Bound voltage magnitudes only at PV and slack buses
    for (i, bus) in ref(pm, :bus)
        bus_type = bus["bus_type"]
        if bus_type == 3 || bus_type == 2
            JuMP.set_lower_bound(vm[i], bus["vmin"])
            JuMP.set_upper_bound(vm[i], bus["vmax"])
        end
    end

    PM.sol_component_value(pm, nw, :bus, :vm, ids(pm, nw, :bus), vm)
    # TODO: check that leaving some voltages unbounded is not leading to unrealistic values.
end

function variable_gen_power_pfdelta(pm; nw::Int=PM.nw_id_default)
    pg = var(pm, nw)[:pg] = JuMP.@variable(pm.model,
        [i in ids(pm, nw, :gen)], base_name="$(nw)_pg",
        start = PM.comp_start_value(ref(pm, nw, :gen, i), "pg_start")
    )

    qg = var(pm, nw)[:qg] = JuMP.@variable(pm.model,
        [i in ids(pm, nw, :gen)], base_name="$(nw)_qg",
        start = PM.comp_start_value(ref(pm, nw, :gen, i), "qg_start")
    )
    
    slack_bus = ids(pm, :ref_buses)

    # Only constraint pg at PV buses. qg is unconstrained for all buses.
    # Slack bus has no constraint in slack generation.

    for (i, gen) in ref(pm, nw, :gen)
        if !(gen["gen_bus"] in slack_bus)
            JuMP.set_lower_bound(pg[i], gen["pmin"])
            JuMP.set_upper_bound(pg[i], gen["pmax"])
        end
    end

    PM.sol_component_value(pm, nw, :gen, :pg, ids(pm, nw, :gen), pg)
    PM.sol_component_value(pm, nw, :gen, :qg, ids(pm, nw, :gen), qg)
end

function variable_branch_power_pfdelta(pm; nw::Int=PM.nw_id_default)
    p = var(pm, nw)[:p] = JuMP.@variable(pm.model,
    [(l,i,j) in ref(pm, nw, :arcs)], base_name="$(nw)_p",
    start = PM.comp_start_value(ref(pm, nw, :branch, l), "p_start")
    )

    q = var(pm, nw)[:q] = JuMP.@variable(pm.model,
    [(l,i,j) in ref(pm, nw, :arcs)], base_name="$(nw)_q",
    start = PM.comp_start_value(ref(pm, nw, :branch, l), "q_start")
    )

    for (l,branch) in ref(pm, nw, :branch)
        if haskey(branch, "pf_start")
            f_idx = (l, branch["f_bus"], branch["t_bus"])
            JuMP.set_start_value(p[f_idx], branch["pf_start"])
        end
        if haskey(branch, "pt_start")
            t_idx = (l, branch["t_bus"], branch["f_bus"])
            JuMP.set_start_value(p[t_idx], branch["pt_start"])
        end
    end

    for (l,branch) in ref(pm, nw, :branch)
        if haskey(branch, "qf_start")
            f_idx = (l, branch["f_bus"], branch["t_bus"])
            JuMP.set_start_value(q[f_idx], branch["qf_start"])
        end
        if haskey(branch, "qt_start")
            t_idx = (l, branch["t_bus"], branch["f_bus"])
            JuMP.set_start_value(q[t_idx], branch["qt_start"])
        end
    end
    
    PM._IM.sol_component_value_edge(pm, PM.pm_it_sym, nw, :branch, :pf, :pt, ref(pm, nw, :arcs_from), ref(pm, nw, :arcs_to), p)   
    PM._IM.sol_component_value_edge(pm, PM.pm_it_sym, nw, :branch, :qf, :qt, ref(pm, nw, :arcs_from), ref(pm, nw, :arcs_to), q)
end
