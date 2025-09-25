#!/usr/bin/env bash
set -euo pipefail

JOB_SCRIPT="jobs/run_close_2inf_samples.sh"   # your SBATCH script
CFG_DIR="data_generation/configs/configs_round1"                  # where your .toml files live

# submit one job per config
for cfg in "$CFG_DIR"/*.toml; do
  [[ -e "$cfg" ]] || { echo "No .toml files in $CFG_DIR"; exit 1; }
  echo "Submitting $cfg"
  sbatch "$JOB_SCRIPT" "$cfg"
done
