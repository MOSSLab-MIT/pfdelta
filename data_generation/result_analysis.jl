using JSON
using Plots
using Statistics


function benchmark_print(benchmark::Dict)
	keys = ["chebyshev", "sample", "acopf", "find_nearest_point", "infeas_cert_and_retry"]
	for key in keys
		avg_time = mean(benchmark[key][2:end])
		println(key, ": ", avg_time)
	end
end


function benchmark_plot(
	key::String="chebyshev";
	xlabel="iteration",
	ylabel="seconds",
	yscale=:identity,
	skip_first=false
)
	results = [time14, time30, time57, time118]
	cases = ["case14", "case30", "case57", "case118"]
	title = key
	plt = plot()
	for (result, case) in zip(results, cases)
		times = result[key]
		if skip_first
			times = times[2:end]
		end
		plot!(times, xlabel=xlabel, ylabel=ylabel,
			plot_title=title, yscale=yscale, label=case)
	end
	display(plt)
end


time14 = JSON.parsefile("experiment_results/time14.json")
time30 = JSON.parsefile("experiment_results/time30.json")
time57 = JSON.parsefile("experiment_results/time57.json")
time118 = JSON.parsefile("experiment_results/time118.json")

println(keys(time14))
