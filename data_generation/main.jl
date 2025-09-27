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
using Logging
using PowerModels

include("src/OPFLearn.jl")
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

cases = Dict(
	"case14" => loadcase("case14"),
	"case30" => loadcase("case30"),
	"case57" => loadcase("case57"),
	"case118" => loadcase("case118"),
	"case500" => loadcase("case500"),
	"case2000" => loadcase("case2000")
)
println("Cases loaded!")

function generator_w_checkpoint(
		folder_path,
		dataset_size,
		point_generator,
		topology_perturb,
)
	progress_results = joinpath(folder_path, "results.json")
	num_parts = 10
	fragment = Int(dataset_size/num_parts)
	A = nothing
	b = nothing
	if ispath(progress_results)
		println("PREVIOUS PROGRESS FOUND!")
		results = JSON.parsefile(progress_results)
	else
		results = Dict()
	end
	for i in 1:num_parts
		if string(i) in keys(results)
			A = hcat(results["A"]...)
			b = reshape(results["b"][1], :, 1)
			println("Part $i/$num_parts already done!")
		else
			initial_k = (i-1) * fragment
			if A !== nothing
				part_results, Anb = point_generator(network, fragment; save_path=folder_path,
					perturb_costs_method="shuffle", perturb_topology_method=topology_perturb,
					starting_k=initial_k, net_path=folder_path, save_max_load=true, A=A, b=b, returnAnb=true)
			else
				part_results, Anb = point_generator(network, fragment; save_path=folder_path,
					perturb_costs_method="shuffle", perturb_topology_method=topology_perturb,
					starting_k=initial_k, net_path=folder_path, save_max_load=true, returnAnb=true)
			end
			A, b = Anb
			part_results["A"] = A
			part_results["b"] = b
			# Save the part metadata
			part_path = joinpath(folder_path, "meta$i.json")
			open(part_path, "w") do io
				JSON.print(io, part_results)
			end
			part_results = nothing
			# Save the data for the next part 
			results["A"] = A
			results["b"] = b
			results[string(i)] = "finished"
			open(progress_results, "w") do io
				JSON.print(io, results)
			end
			println("\n\n\n##################################################")
			println("##\t\t\t\t\t\t\t\t##")
			println("##\t\tCREATED PART $i/$(num_parts)!\t\t\t##")
			println("##\t\t\t\t\t\t\t\t##")
			println("##################################################\n\n\n")
		end
	end
end

# Entry point
if length(ARGS) == 0
	println("No arguments provided!")
else # 1st linear/parallel, 2nd case name, 3rd topology perturbation
	# data_method, comp_method = split(ARGS[1], "_")
	comp_method = ARGS[1]
	network_name = ARGS[2]
	topology_perturb = ARGS[3]
	data_dir = ARGS[4] # setting an absolute path here for now.

	# Handles comp_method
	point_generator = nothing
	if comp_method == "linear"
		point_generator = OPFLearn.create_samples
		parallel = false
	elseif comp_method == "parallel"
		point_generator = OPFLearn.dist_create_samples
		parallel = true
	end

	# Handles checkpoints
	with_checkpoint = nothing
	if length(ARGS) >= 5
		with_checkpoint = true
	else
		with_checkpoint = false
	end

	# Loads correct network
	network = cases[network_name]

	# Sets topology perturbation and number of samples for it
	dataset_size = nothing
	if topology_perturb == "none"
		dataset_size = 56000
	elseif topology_perturb == "n-1"
		dataset_size = 29000
	else
		dataset_size = 20000
	end

	println("Doing case: $network_name, perturbation: $topology_perturb, comp method: $comp_method")

	# Set up folders
	folder_path = joinpath(data_dir, network_name, topology_perturb)
	mkpath(joinpath(folder_path, "raw"))

	# Start generation
	if !with_checkpoint
		# Create samples
		results = point_generator(network, dataset_size; save_path=folder_path, perturb_costs_method="shuffle",
			perturb_topology_method=topology_perturb, net_path=folder_path, save_max_load=true,)
		# Results file name
		file_name = joinpath(folder_path, "results.json")
		open(file_name, "w") do io
			JSON.print(io, results)
		end
	else
		generator_w_checkpoint(
			folder_path,
			dataset_size,
			point_generator,
			topology_perturb,
		)
	end

	println("DONE")

end
