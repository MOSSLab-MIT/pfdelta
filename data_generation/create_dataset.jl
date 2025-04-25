using LinearAlgebra
using Graphs

include("graph_creator.jl")
include("src/OPFLearn.jl")

function dev_to_euc(network, deviation)
	# Gather loads
	loads = network["load"]
	Pd = [loads[string(i)]["pd"] for i in 1:length(loads)]
	Qd = [loads[string(i)]["qd"] for i in 1:length(loads)]
	Sd = Pd .+ Qd .* im
	# Calculate the largest deviation in a +- %deviation
	dev_Sd = Sd .* deviation
	# Calculate L2 norm of this deviation, multiply by 2 to guarantee nonoverlap
	min_distance = sqrt(sum(abs.(dev_Sd).^2)) * 2
	return min_distance
end


function create_dataset_seeds(
	network::Dict,
	num_seeds::Integer;
	starting_distance::Float64=-1.0,
	min_distance::Float64=-1.0,
	exp_reduction::Float64=0.9,
	portion_of_new_seeds::Float64=0.1,
	file_name::String="experiment.json",
	point_generator=OPFLearn.create_samples
)
	println("Producing $num_seeds good seeds!")
	# Gather initial set
	results, Anb = point_generator(network, num_seeds; returnAnb=true)
	print("\n\n\n")
	A, b = Anb
	println("Initial seeds produced.")
	P = results["inputs"]["pd"]
	Q = results["inputs"]["qd"]
	seeds = P .+ Q .* im
	# Save results
	all_results = Dict("initial" => results)
	open(file_name, "w") do io
		JSON.print(io, all_results)
	end

	# Set up starting_distance default value
	starting_distance < 0 && (starting_distance = Inf)

	# Create initial graph
	g, g_weights = create_graph(seeds, starting_distance)
	g_edges = collect(keys(g_weights))
	distances = collect(values(g_weights))

	# Set up more default values
	max_distance = maximum(distances)
	min_distance == -1. && (min_distance = minimum(distances) - 0.0001)
	min_distance == -2. && (min_distance = dev_to_euc(network, 0.05))

	println("$num_seeds samples generated! Will see if they are enough")
	# Start iterating
	if starting_distance != Inf
		max_radius = starting_distance
	else
		max_radius = max_distance
	end
	println("Currently trying radius: $max_radius")
	counter = 0
	while true
		flush(stdout)
		# First, prune the graph slowly to make it tractable for MIS
		# Also, it is unlikely that this many edges is conducive to 
		# a small MIS
		println("Minimum radius: $min_distance")
		if ne(g) > nv(g) * 10
			println("Manually prunning graph to make MIS tractable " *
				"and to make chances of a big MIS more likely!")
			while length(g_edges) > nv(g) * 10
				println("Radius of $max_radius gave ", nv(g), " nodes and ", length(g_edges), " edges. ")
				max_radius *= 0.9
				mask = distances .< max_radius
				distances = distances[mask]
				g_edges = g_edges[mask]
			end
			print("Will use radius of $max_radius for next step. ")
			# Recreate graph with subset of edges
			g = SimpleGraph(nv(g))
			for (u, v) in Tuple.(g_edges)
				add_edge!(g, u, v)
			end
		end
		# We reduce the max_radius slowly until it hits the right number
		println("Prunning graph by using MIS now...")
		if max_radius <= min_distance
			println("Radius already too small!")
		end
		while max_radius > min_distance
			# Calculate if max_radius is good
			candidate_seeds = find_maximum_independent_set(g)
			if length(candidate_seeds) > num_seeds
				println(
					"Success! $num_seeds good seeds produced!"
				)
				println("Successful radius: $max_radius")
				println("Saving them...")
				ds_results = Dict(
					"good_seeds" => candidate_seeds,
					"all_seeds" => seeds,
					"max_radius" => max_radius
				)
				all_results["ds_results"] = ds_results
				open(file_name, "w") do io
					JSON.print(io, all_results)
				end
				return all_results
			end
			# Otherwise, reduce radius
			MIS_size = length(candidate_seeds)
			max_radius *= 0.9
			println("Current radius gives small MIS of size $MIS_size. " *
				"Now trying radius: $max_radius")
			mask = distances .> max_radius # True => remove edge
			distances = distances[.!mask] # keep these
			bad_edges = g_edges[mask] # remove these
			g_edges = g_edges[.!mask] # keep these
			for (u, v) in Tuple.(bad_edges)
				rem_edge!(g, u, v)
			end
		end
		num_seeds_produced = size(seeds, 1)
		println("Current $num_seeds_produced" * 
			" not enough! Producing more seeds")
		# If it is not enough, create more seeds, reset max_radius
		num_new_seeds = floor(Int, num_seeds * portion_of_new_seeds)
		new_results, (A, b) = point_generator(
			network, num_new_seeds; returnAnb=true, A=A, b=b)
		print("\n\n\n")
		P = new_results["inputs"]["pd"]
		Q = new_results["inputs"]["qd"]
		new_seeds = P .+ Q .* im
		println("Produced $num_new_seeds new seeds! Checking if enough...")
		# Add them to the set of points
		seeds = vcat(seeds, new_seeds)
		g, g_weights = create_graph(seeds, starting_distance)
		g_edges = collect(keys(g_weights))
		distances = collect(values(g_weights))
		# Reset radius to try
		if starting_distance != Inf
			max_radius = starting_distance
		else
			max_radius = max_distance
		end
		println("Max radius reset to $max_radius")
		# Save progress
		all_results[string(counter)] = new_results
		open(file_name, "w") do io
			JSON.print(io, all_results)
		end
		counter += 1
	end
end
