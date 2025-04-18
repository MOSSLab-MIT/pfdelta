using PowerModels
using Statistics
using Plots
using Debugger


include("src/OPFLearn.jl")


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

function benchmark_print(benchmark::Dict)
	keys = ["chebyshev", "sample", "acopf", "find_nearest_point", "infeas_cert_and_retry"]
	for key in keys
		avg_time = mean(benchmark[key][2:end])
		println(key, ": ", avg_time)
	end
end


N = 1000
case14 = loadcase("case14")
# case30 = loadcase("case30")
# case57 = loadcase("case57")

_, time14 = OPFLearn.create_samples(case14, N)
# _, time30 = OPFLearn.create_samples(case30, N)
# _, time57 = OPFLearn.create_samples(case57, N)

