using LinearAlgebra
using Graphs
using JuMP
using HiGHS


function pairwise_distances(X::AbstractMatrix{T}) where T<:Number
    G = X * X'  # This automatically uses Hermitian (conjugate transpose)
    norms = real.(diag(G))
    D_squared = @. norms + norms' - 2 * real(G)
    D = sqrt.(max.(D_squared, 0.0))  # guard against tiny negative values
    return D
end

"This method takes inspiration in Graphs.SimpleGraphs.euclidean_graphs"
function create_graph(
	points::AbstractMatrix{T},
	threshold::Float64=Inf
) where T<:Number
	num_points = size(points, 1)
	distances = pairwise_distances(points)
	weights = Dict{Graphs.SimpleEdge{Int}, Float64}()

	for point in 1:num_points
		for index in point+1:num_points
			if distances[point, index] < threshold
				e = Graphs.SimpleEdge(point, index)
				weights[e] = distances[point, index]
			end
		end
	end
	g = Graphs.SimpleGraphs._SimpleGraphFromIterator(keys(weights), Int)
	if nv(g) < num_points
		add_vertices!(g, num_points - nv(g))
	end
	return g, weights
end


function find_maximum_independent_set(
	g::SimpleGraph{Int64};
	optimizer=HiGHS.Optimizer,
)
	# Initialize model
	model = JuMP.Model(optimizer)
	set_optimizer_attribute(model, "output_flag", false)

	# Create binary variables for vertices
	@variable(model, vertices[1:nv(g)], Bin)

	# Implement constraint for no edge being present
	for e in edges(g)
		u, v = src(e), dst(e)
		@constraint(model, vertices[u] + vertices[v] <= 1)
	end

	# Create objective to maximize number of vertices chosen
	@objective(model, Max, sum(vertices))

	# Find independent set
	optimize!(model)

	# Extract solution
	independent_set = [v for v in 1:nv(g) if value(vertices[v]) > 0.5]

	return independent_set

end
