using Pkg
Pkg.activate(".")
using Debugger 
import PowerModels
import PowerModels: ref, var, ids
const PM = PowerModels
import JuMP
import Ipopt
import JSON
include("src/build_opf_pfdelta.jl")
include("src/pf_delta_perturbations.jl")

# network = PM.parse_file("pglib/pglib_opf_case5_pjm.m")
# network["gen"]["1"]["gen_status"] = 0
# network["gen"]["2"]["gen_status"] = 0
sample_path = "final_data_no_exp/case500/n-2/raw/sample_7933.json"
sample_data = JSON.parsefile(sample_path)
network = sample_data["network"]


change_bus_type!(network)
results = PM.compute_ac_pf(network, flat_start=true)


