import os

data_dir = "/home/akrivera/orcd/scratch/pfdelta_data"
delete_intermediate = True
analysis_mode = False

cases = ["case14", "case30", "case57", "case118", "case500", "case2000"]
topologies = ["n", "n-1", "n-2"]

matlab_home = ""  # leave empty on cluster
matpower_path = "/orcd/home/002/akrivera/third_party/matpower/matpower8_1/"

# Template for the config file
TEMPLATE = """data_dir = "{data_dir}"
case_name = "{case_name}"
topology_perturb = "{topology}"

delete_intermediate = {delete_intermediate}
analysis_mode = {analysis_mode}

[matlab]
# Note: Code has only been tested on this MATLAB version
# TODO: you need to add info here on which matpower version should be used too.
# TODO: add note on intel matlab only for mac
# TODO: add note to leave empty on cluster
home = "{matlab_home}"
matpower_path = "{matpower_path}"
"""

outdir = "."
os.makedirs(outdir, exist_ok=True)

for case in cases:
    for topo in topologies:
        filename = f"{case}_{topo}.toml"
        content = TEMPLATE.format(
            data_dir=data_dir,
            case_name=case,
            topology=topo,
            delete_intermediate=str(delete_intermediate).lower(),
            analysis_mode=str(analysis_mode).lower(),
            matlab_home=matlab_home,
            matpower_path=matpower_path,
        )
        with open(os.path.join(outdir, filename), "w") as f:
            f.write(content)

print(f"✅ Generated {len(cases) * len(topologies)} config files in '{outdir}/'")
