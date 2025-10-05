#!/usr/bin/env bash
#SBATCH -p mit_normal
#SBATCH -J test_matlab_jl
#SBATCH -c 2
#SBATCH --mem=32G
#SBATCH -t 00:05:00
#SBATCH -o logs/test_matlab_jl.%j.out
#SBATCH -e logs/test_matlab_jl.%j.err
set -euo pipefail

module purge
module load julia/1.9.1
module load matlab/matlab-2024a

# If MATLAB.jl needs it, infer ROOT from the module's matlab on PATH:
MATLAB_BIN="$(command -v matlab)"
if [[ -n "${MATLAB_BIN}" ]]; then
  export MATLAB_ROOT="$(dirname "$(dirname "$MATLAB_BIN")")"
fi

# Check licenses/path
which matlab
matlab -batch "ver; exit"

# Check MATLAB.jl roundtrip
julia --project=. -e 'using Pkg; Pkg.add("MATLAB"); using MATLAB; mat"disp(version)"; mat"ver"'

echo "✅ Job finished successfully at $(date)"
