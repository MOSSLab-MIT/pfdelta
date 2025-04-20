using LinearAlgebra
using Graphs

include("graph_creator.jl")
include("src/OPFLearn.jl")


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
	# Gather initial set
	results, _ = point_generator(network, num_seeds)
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
	min_distance < 0 && (min_distance = minimum(distances))

	println("$num_seeds samples generated! Will see if they are enough")
	# Start iterating
	if starting_distance != Inf
		max_radius = starting_distance
	else
		max_radius = max_distance
	end
	counter = 0
	while true
		# First, prune the graph slowly to make it tractable for MIS
		if ne(g) > nv(g) * 10
			while length(distances) > nv(g) * 10
				max_radius *= 0.9
				mask = distances .< max_radius
				distances = distances[mask]
				g_edges = g_edges[mask]
			end
			g = Graphs.SimpleGraphs._SimpleGraphFromIterator(g_edges)
			if nv(g) < size(seeds, 1)
				add_vertices!(g, size(seeds, 1) - nv(g))
			end
		end
		# We reduce the max_radius slowly until it hits the right number
		while max_radius > min_distance
			# Calculate if max_radius is good
			candidate_seeds = find_maximum_independent_set(g)
			if length(candidate_seeds) > num_seeds
				println(
					"Success! $num_seeds seeds produced!"
				)
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
				return ds_results
			end
			# Otherwise, reduce radius
			max_radius *= 0.9
			mask = distances .< max_radius
			distances = distances[mask]
			g_edges = g_edges[mask]
			g = Graphs.SimpleGraphs._SimpleGraphFromIterator(g_edges)
			if nv(g) < size(seeds, 1)
				add_vertices!(g, size(seeds, 1) - nv(g))
			end
		end
		num_seeds = size(seeds, 1)
		println("Current $num_seeds not enough! Producing more seeds")
		# If it is not enough, create more seeds
		num_new_seeds = floor(Int, num_seeds * portion_of_new_seeds)
		new_results, _ = point_generator(network, num_new_seeds)
		P = new_results["inputs"]["pd"]
		Q = new_results["inputs"]["qd"]
		new_seeds = P .+ Q .* im
		println("Produced $num_new_seeds new seeds! Checking if enough...")
		# Add them to the set of points
		seeds = vcat(seeds, new_seeds)
		g, g_weights = create_graph(seeds, starting_distance)
		if starting_distance != Inf
			max_radius = starting_distance
		else
			max_radius = max_distance
		end
		# Save progress
		all_results[string(counter)] = new_results
		open(file_name, "w") do io
			JSON.print(io, all_results)
		end
		counter += 1
	end
end
