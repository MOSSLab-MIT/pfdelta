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
	results, time = OPFLearn.create_samples(case57, 750)
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
	all_results = create_dataset_seeds(
		case14, 7500; min_distance=0.3, file_name="alg14.json")
	json_str = JSON.json(dict; indent=3)
	print(json_str)
elseif ARGS[1] == "algorithm30"
	all_results = create_dataset_seeds(
		case30, 5000; min_distance=0.3, file_name="alg30.json")
	json_str = JSON.json(dict; indent=3)
	print(json_str)
elseif ARGS[1] == "algorithm57"
	all_results = create_dataset_seeds(
		case57, 500; min_distance=0.3, file_name="alg57.json")
	json_str = JSON.json(dict; indent=3)
	print(json_str)
elseif ARGS[1] == "algorithm118"
	all_results = create_dataset_seeds(
		case118, 50; min_distance=0.3, file_name="alg118.json")
	json_str = JSON.json(dict; indent=3)
	print(json_str)
elseif ARGS[1] == "parallel14"
	results = create_dataset_seeds(case14, 10000;
		min_distance=-2., point_generator=OPFLearn.dist_create_samples, file_name="case14_data/seeds.json")
elseif ARGS[1] == "parallel30"
	results = create_dataset_seeds(case30, 5000;
		min_distance=-2., point_generator=OPFLearn.dist_create_samples, file_name="case30_data/seeds.json")
elseif ARGS[1] == "parallel57"
	results = create_dataset_seeds(case57, 1000;
		min_distance=-2., point_generator=OPFLearn.dist_create_samples, file_name="case57_data/seeds.json")
elseif ARGS[1] == "parallel118"
	results = create_dataset_seeds(case118, 500;
		min_distance=-2., point_generator=OPFLearn.dist_create_samples, file_name="case118_data/seeds.json")
end
