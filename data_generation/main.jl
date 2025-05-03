using Distributed

# Activate project on all separately, step by step
@everywhere begin
using Pkg
Pkg.activate(".")
end

@everywhere begin
using JSON
using Debugger
using Plots
using Statistics

using PowerModels

include("src/OPFLearn.jl")
include("create_dataset.jl")
end

function loadcase(casenum::String)
	case_to_path = Dict(
		"case14" => "pglib/pglib_opf_case14_ieee.m",
		"case30" => "pglib/pglib_opf_case30_ieee.m",
		"case57" => "pglib/pglib_opf_case57_ieee.m",
		"case118" => "pglib/pglib_opf_case118_ieee.m",
		"case500" => "pglib/pglib_opf_case500_goc.m",
		"case2000" => "pglib/pglib_opf_case2000_goc.m"
	)
	path = case_to_path[casenum]
	network = PowerModels.parse_file(path)

	return network
end

case14 = loadcase("case14")
case30 = loadcase("case30")
case57 = loadcase("case57")
case118 = loadcase("case118")
cases = Dict(
	"case14" => case14,
	"case30" => case30,
	"case57" => case57,
	"case118" => case118,
	"case500" => loadcase("case500"),
	"case2000" => loadcase("case2000")
)
println("Cases loaded!")

if length(ARGS) == 0
	println("No argument!")
elseif ARGS[1] == "case14"
	results, time = OPFLearn.create_samples(case14, 10000)
	open("time14.json", "w") do io
	    JSON.print(io, time)
	end
	open("case14.json", "w") do io
	    JSON.print(io, results)
	end
elseif ARGS[1] == "case30"
	results, time = OPFLearn.create_samples(case30, 5000)
	open("time30.json", "w") do io
	    JSON.print(io, time)
	end
	open("results30.json", "w") do io
	    JSON.print(io, results)
	end
elseif ARGS[1] == "case57"
	results, time = OPFLearn.create_samples(case57, 5600)
	open("time57.json", "w") do io
	    JSON.print(io, time)
	end
	open("results57.json", "w") do io
	    JSON.print(io, results)
	end
elseif ARGS[1] == "case118"
	results, time = OPFLearn.create_samples(case118, 50)
	open("time118.json", "w") do io
	    JSON.print(io, time)
	end
	open("results118.json", "w") do io
	    JSON.print(io, results)
	end
else # 1st linear/parallel, 2nd case name, 3rd topology perturbation
	comp_method = ARGS[1]
	network_name = ARGS[2]
	topology_perturb = ARGS[3]
	if comp_method == "linear"
		point_generator = OPFLearn.create_samples
		parallel = false
	elseif comp_method == "parallel"
		point_generator = OPFLearn.dist_create_samples
		parallel = true
	end
	network = cases[network_name]
	if topology_perturb == "none"
		dataset_size = 56000
	elseif topology_perturb == "n-1"
		dataset_size = 29000
	else
		dataset_size = 20000
	end
	seeds_needed = trunc(Int, dataset_size * 0.03)
	samples_per_seed = ceil(Int, dataset_size / seeds_needed) - 1
	folder_path = joinpath("$(network_name)_seeds", topology_perturb)
	if network_name == "case57"
		portion_of_new_seeds = 0.3
	else
		portion_of_new_seeds = 0.1
	end
	println("Doing case: $network_name, perturbation: $topology_perturb, and comp method: $comp_method")
	allseeds = create_dataset_seeds(
		network, seeds_needed; perturb_topology_method=topology_perturb, perturb_costs_method="shuffle",
		min_distance=-2., save_path=folder_path, portion_of_new_seeds=portion_of_new_seeds)
	println("\n\n\n#########################\n\n\n")
	expand_dataset_seeds(
		joinpath(folder_path, "seeds.json"), samples_per_seed; base_case=network,
		cp_seeds_to_raw=true, seed_expander=OPFLearn.create_seed_samples,
		perturb_topology_method=topology_perturb, perturb_costs_method="shuffle",
		parallel=parallel
	)
end
