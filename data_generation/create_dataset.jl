using LinearAlgebra
using Graphs

include("graph_creator.jl")
include("src/OPFLearn.jl")

function dev_to_euc(network, deviation)
	# Gather loads
	loads = network["load"]
	Pd = [loads[string(i)]["pd"] for i in 1:length(loads)]
	# Calculate the largest deviation in a +- %deviation
	dev_Pd = Pd .* deviation
	# Calculate L2 norm of this deviation, multiply by 2 to guarantee nonoverlap
	min_distance = sqrt(sum(dev_Pd.^2)) * 2
	return min_distance
end


function create_dataset_seeds(
	network::Dict,
	num_seeds::Integer;
	perturb_topology_method::String, # Options: none, n-1, n-2
	perturb_costs_method::String, # Options: shuffle, none
	starting_distance::Float64=-1.0,
	min_distance::Float64=-1.0,
	exp_reduction::Float64=0.9,
	portion_of_new_seeds::Float64=0.1,
	save_path::String="experiment.json",
	point_generator=OPFLearn.create_samples,
)
	# Create folder for data
	folder_name = save_path
	file_name = joinpath(folder_name, "seeds.json")
	mkpath(folder_name)
	# If min distance = -2, it can be done early
	min_distance == -2. && (min_distance = dev_to_euc(network, 0.05))
	println("HERE IS THE MIN DISTANCE: ", min_distance)
	# Start seed production
	println("Producing $num_seeds good seeds!")
	# Make seed folder if needed
	seeds_storage_path = joinpath(save_path, "allseeds")
	if !isdir(seeds_storage_path)
		mkpath(seeds_storage_path)
	end
	# Gather initial set
	results, Anb = point_generator(network, num_seeds; returnAnb=true, save_path=save_path,
		perturb_costs_method=perturb_costs_method, perturb_topology_method=perturb_topology_method)
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
	g, g_weights = create_graph(real(seeds), starting_distance)
	g_edges = collect(keys(g_weights))
	distances = collect(values(g_weights))

	# Set up more default values
	max_distance = maximum(distances)
	min_distance == -1. && (min_distance = minimum(distances) - 0.0001)

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
				if max_radius * 0.9 < min_distance
					println("Radius cannot be reduced anymore without falling behind min! " *
						"Fingers crossed...")
					break
				end
				max_radius *= 0.9
				mask = distances .< max_radius
				distances = distances[mask]
				g_edges = g_edges[mask]
			end
			println("Will use radius of $max_radius for next step.")
			# Recreate graph with subset of edges
			g = SimpleGraph(nv(g))
			for (u, v) in Tuple.(g_edges)
				add_edge!(g, u, v)
			end
		end
		# We reduce the max_radius slowly until it hits the right number
		println("Currently there are $(nv(g)) nodes and $(ne(g)) edges. Prunning graph by using MIS now...")
		if max_radius <= min_distance
			println("Radius already too small!")
		end
		while max_radius > min_distance
			# Calculate if max_radius is good
			candidate_seeds = find_maximum_independent_set(g)
			if length(candidate_seeds) >= num_seeds
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
		new_results, (A, b) = point_generator(network, num_new_seeds; returnAnb=true, A=A, b=b,
			save_path=save_path, perturb_costs_method=perturb_costs_method,
			perturb_topology_method=perturb_topology_method, starting_k=num_seeds_produced)
		print("\n\n\n")
		P = new_results["inputs"]["pd"]
		Q = new_results["inputs"]["qd"]
		new_seeds = P .+ Q .* im
		println("Produced $num_new_seeds new seeds! Checking if enough...")
		# Add them to the set of points
		seeds = vcat(seeds, new_seeds)
		g, g_weights = create_graph(real(seeds), starting_distance)
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


function expand_dataset_seeds(
	seed_location::String,
	samples_per_seed::Integer;
	base_case::Dict,
	cp_seeds_to_raw::Bool=false,
	seed_expander=nothing,
	perturb_topology_method::String,
	perturb_costs_method::String,
	parallel::Bool=false,
)
	println("Will expand seeds to full dataset!\n\n")
	println("Seeds being expanded: $seed_location")
	println("Expanding each seed to $samples_per_seed samples")
	# Gather seed information
	seeds_dict = JSON.parsefile(seed_location)
	good_seeds = seeds_dict["ds_results"]["good_seeds"]
	num_seeds = length(good_seeds)
	println("Expanding $num_seeds seeds")
	sampling_radius = seeds_dict["ds_results"]["max_radius"] / 2
	# Collect good seeds in raw folder
	folder_path = dirname(seed_location)
	raw_path = joinpath(folder_path, "raw")
	println("Copy seeds from seed folder to raw: $cp_seeds_to_raw")
	if cp_seeds_to_raw
		mkpath(raw_path)
		allseeds_raw_path = joinpath(folder_path, "allseeds")
		for (i, real_id) in zip(good_seeds, 1:num_seeds)
			seed_raw_path = joinpath(allseeds_raw_path, "sample_$i.json")
			new_path = joinpath(raw_path, "sample_$real_id.json")
			cp(seed_raw_path, new_path)
		end
		println("Seeds copied!")
	end
	# Initialize dictionary to keep track of seed origin, init path too
	seed_origin = Dict{Int, Vector{Integer}}()
	seed_origin_path = joinpath(folder_path, "seed_origin.json")
	# Separate cases for parallel and sequential
	if parallel
		seed_origin_data = @distributed (vcat) for i in 1:num_seeds
				origin_datum = expand_one_seed(i, raw_path, samples_per_seed, num_seeds, seed_expander,
				sampling_radius, perturb_topology_method, perturb_costs_method, base_case
			)
			[origin_datum]
		end
	else
		# Keep track of seed origin data
		seed_origin_data = []
		# Produce samples
		for i in 1:num_seeds
			origin_datum = expand_one_seed(i, raw_path, samples_per_seed, num_seeds, seed_expander,
				sampling_radius, perturb_topology_method, perturb_costs_method, base_case
			)
			push!(seed_origin_data, origin_datum)
		end
	end
	println("Seed expansion has ended! Saving seed origin dictionary and closing...")
	# Create seed origin and save it
	for datum in seed_origin_data
		i, i_range = datum
		seed_origin[i] = i_range
	end
	open(seed_origin_path, "w") do io
		JSON.print(io, seed_origin)
	end
end


function expand_one_seed(i, raw_path, samples_per_seed, num_seeds, seed_expander,
	sampling_radius, perturb_topology_method, perturb_costs_method, base_case
)
	println("### EXPANDING SEED $i / $num_seeds")
	# Load pm dict
	raw_seed_path = joinpath(raw_path, "sample_$i.json")
	raw_dict = JSON.parsefile(raw_seed_path)
	# Set starting point according to the other seeds produced
	starting_id = num_seeds + ((i-1)*samples_per_seed)
	# Produce samples
	seed_expander(
		net=raw_dict["network"],
		radius=sampling_radius,
		num_samples=samples_per_seed,
		base_case=base_case,
		path_to_save=raw_path,
		starting_id=starting_id,
		seed_id=i,
		perturb_topology_method=perturb_topology_method,
		perturb_costs_method=perturb_costs_method
	)
	return (i, starting_id:starting_id+samples_per_seed)
end
