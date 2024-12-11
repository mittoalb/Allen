#!/bin/bash

# Check if the correct number of arguments is provided
if [ "$#" -lt 3 ]; then
    echo "Usage: $0 <num_chunks> <num_projection> <command>"
    exit 1
fi

# Extract the number of chunks and the command
num_chunks=$1
total_projs=$2
shift 2
command="$@"

# Calculate the chunk size
chunk_size=$((total_projs / num_chunks))
# Loop through the chunks and execute the command
for ((i=0; i<num_chunks; i++)); do
    start_proj=$((i * chunk_size))
    end_proj=$((start_proj + chunk_size))
    echo "Executing chunk $((i + 1)): --start-proj $start_proj --end-proj $end_proj"
    eval "$command --start-proj $start_proj --end-proj $end_proj"
    mv _rec _rec/_rec_$i
done

