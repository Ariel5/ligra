#!/bin/bash

# Define the directories
ys_dir="/home/ubuntu/prog/erdos-renyi-10-degree/Ys"
ligra_dir="/home/ubuntu/prog/erdos-renyi-10-degree/Ligra-AdjGraphs"

# Create an array of files sorted by the number of nodes
readarray -t sorted_files < <(find "$ligra_dir" -name "*.AdjGraph" | sort -t '_' -k1,1n)

# Loop through the sorted array of files
for ligra_file in "${sorted_files[@]}"; do
  # Extract the base name without the extension
  base_name=$(basename "$ligra_file" .AdjGraph)

  # Check if the corresponding Ys file exists
  ys_file="$ys_dir/${base_name}.csv"
  if [[ -f "$ys_file" ]]; then
    # Print the name of the graph
    echo "Processing graph: $base_name"

    # Run your command with the Ys file and the Ligra file
    ./GraphEncoderEmbedding -rounds 7 -nClusters 50 -saveEmbedding false -Laplacian false -yLocation "$ys_file" "$ligra_file"
  else
    echo "Warning: Ys file for $base_name does not exist."
  fi
done
