using Pkg
Pkg.activate(".")
Pkg.instantiate()

using PowerModels
using Statistics
using Plots
using Debugger
using JSON


include("src/OPFLearn.jl")
include("create_dataset.jl")


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
	results, time = OPFLearn.create_samples(case57, 250)
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
elseif ARGS[1] == "algorithm14"
	all_results = create_dataset_seeds(case14, 1000, min_distance=0.3)
	json_str = JSON.json(dict; indent=3)
	print(json_str)
end
