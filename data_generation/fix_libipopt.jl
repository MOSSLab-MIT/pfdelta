# fix_libipopt.jl
#
# Script to resolve the "libipopt not defined" error in Julia
# Removes broken Ipopt builds and reinstalls from scratch
# Works for Julia 1.9 

using Pkg
import Base.Filesystem: rm, joinpath, isdir

println("🚨 Removing existing Ipopt installation and cleaning up...")
Pkg.rm("Ipopt")
Pkg.gc()

println("🧹 Removing cached artifacts and compiled files...")
home = ENV["HOME"]
julia_dir = joinpath(home, ".julia")
paths = [
    joinpath(julia_dir, "artifacts"),
    joinpath(julia_dir, "compiled"),
    joinpath(julia_dir, "packages", "Ipopt")
]

for path in paths
    if isdir(path)
        println("Deleting ", path)
        try
            rm(path; force=true, recursive=true)
        catch e
            @warn "Failed to delete $path" exception=(e, catch_backtrace())
        end
    end
end

println("📦 Re-adding Ipopt package...")
Pkg.add("Ipopt")

println("🔧 Rebuilding Ipopt...")
Pkg.build("Ipopt")

println("✅ Ipopt fix complete. Restart julia and test with:")
println()
println("    using JuMP, Ipopt")
println("    model = Model(Ipopt.Optimizer)")
println("    @variable(model, x)")
println("    @objective(model, Min, (x - 3)^2)")
println("    optimize!(model)")
println("    println(\"Optimal x = \", value(x))")
