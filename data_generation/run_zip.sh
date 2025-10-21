#!/bin/bash
# Pack each case folder into case.tar.gz

# List of casenames
cases=(
    # case14
    # case30
    # case57
    # case118
    # case500
    case2000
)

for case in "${cases[@]}"; do
    if [[ ! -d "$case" ]]; then
        echo "⚠️ Directory $case not found, skipping."
        continue
    fi

    archive="${case}.tar.gz"
    echo "Creating $archive from $case/"
    tar -czf "$archive" "$case"
done
