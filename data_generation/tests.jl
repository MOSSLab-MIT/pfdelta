using Pkg
Pkg.activate(@__DIR__)

using JSON
using Debugger
include("src/OPFLearn.jl")

network = "pglib/pglib_opf_case14_ieee.m"
dataset_size = 50
perturb_costs_method= "shuffle"
topology_perturb= "n-1"
net_path = "."
save_max_load= true
folder_path = joinpath("case14", topology_perturb) # TODO: rename to raw
mkpath(joinpath(folder_path, "allseeds"))

point_generator = OPFLearn.create_samples
results = @enter point_generator(network, dataset_size; save_path=folder_path, perturb_costs_method="shuffle",
                        perturb_topology_method=topology_perturb, net_path=net_path, save_max_load=true)

# why does net_path work with folder_path?

# Save results JSON
file_name = joinpath(folder_path, "results.json")
open(file_name, "w") do io
    JSON.print(io, results)
end