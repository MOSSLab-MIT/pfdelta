using Pkg
Pkg.activate(".")
using Debugger 
import PowerModels
import PowerModels: ref, var, ids
const PM = PowerModels
import JuMP
import Ipopt
import JSON
include("src/build_opf_pfdelta.jl")
include("src/pf_delta_perturbations.jl")

function change_bus_type!(network)
    bus_gens = Dict{Int, Vector{Dict{String,Any}}}()
    for (_, gen) in network["gen"]
        bus = gen["gen_bus"]
        push!(get!(bus_gens, bus, Vector{Dict{String,Any}}()), gen)
    end
    
    for (_, bus) in network["bus"]
        bus_idx = bus["bus_i"]

        if !haskey(bus_gens, bus_idx) || all(gen["gen_status"] == 0 for gen in bus_gens[bus_idx])
            bus["bus_type"] = 1 # became a PQ bus
        end
    end
end

# network = PM.parse_file("pglib/pglib_opf_case5_pjm.m")
# network["gen"]["1"]["gen_status"] = 0
# network["gen"]["2"]["gen_status"] = 0
sample_path = "case14/n-1/allseeds/sample_2.json"
sample_data = JSON.parsefile(sample_path)
network = sample_data["network"]


change_bus_type!(network) # NOTE: with the new fixes this shouldn't be needed anymore!
results = PM.compute_ac_pf(network, flat_start=true)


