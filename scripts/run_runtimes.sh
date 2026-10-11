#!/bin/bash
#SBATCH --job-name=runtimes_all_flat
#SBATCH --time=20:00:00
#SBATCH --cpus-per-task=1
#SBATCH --output=%x-%j.out

# Flat-start runtime sweep.
# Runs feasible (raw) and near-infeasible (nose) for each case.
# Args to main_runtimes.jl: <case_name> <flat_start> <near_infeasible_flag>

set -uo pipefail

CASES=(case57 case118 case500)
FLAT_START=true            # flat-start runs

failures=()

for case in "${CASES[@]}"; do
    for near_infeasible in false true; do
        echo "=== $(date '+%F %T') | $case | flat_start=$FLAT_START | near_infeasible=$near_infeasible ==="
        if ! julia scripts/main_runtimes.jl "$case" "$FLAT_START" "$near_infeasible"; then
            echo "!! FAILED: $case (near_infeasible=$near_infeasible)"
            failures+=("$case/near_infeasible=$near_infeasible")
        fi
    done
done

if [ ${#failures[@]} -gt 0 ]; then
    echo "=== Completed with ${#failures[@]} failure(s): ${failures[*]} ==="
    exit 1
fi
echo "=== All runs completed successfully ==="
