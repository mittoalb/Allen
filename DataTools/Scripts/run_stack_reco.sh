#!/bin/bash

# Define the path to search for directories
#use: ./run_stack_reco.sh data_path COR type (double_fov, normal) rec_type (full, try)
path=$1
COR=$2
type=$3
rec_type=$4

# List all directories in the specified path at the first level
directories=$(find "$path" -mindepth 1 -maxdepth 1 -type d)


for dir in $directories; do
    	subfolder=$(find "$dir" -mindepth 1 -maxdepth 1 -type d)
	cd "$subfolder" || continue  # Skip if unable to enter the subfolder
	pwd
	# Print the current directory
	echo "Executing commands in: $subfolder"
	python ../../../../../../ESRF2APS.py -i *h5 -o APS.h5
	tomocupy recon  --remove-stripe-method fw --file-name APS.h5 --reconstruction-type $rec_type --nsino-per-chunk 2 --file-type $type --rotation-axis $COR --rotation-axis-method vo --rotation-axis-auto manual
	echo $subfolder "processed"
	cd - > /dev/null  # Return to previous directory silently
done

