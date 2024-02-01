#!/bin/bash

# Define the directories
ys_dir="/home/ubuntu/prog/erdos-renyi-10-degree/Ys"
ligra_dir="/home/ubuntu/prog/erdos-renyi-10-degree/Ligra-AdjGraphs"

# Loop through the Ys directory
for ys_file in "$ys_dir"/*.csv; do
  # Extract the base name without the extension
  base_name=$(basename "$ys_file" .csv)

  # Construct the path to the corresponding Ligra file
  ligra_file="$ligra_dir/${base_name}.AdjGraph"

  echo "Processing graph: $base_name"
  # Run your command with the Ys file and the Ligra file
  ./GraphEncoderEmbedding -rounds 7 -nClusters 50 -saveEmbedding false -Laplacian false -yLocation "$ys_file" "$ligra_file"
done
