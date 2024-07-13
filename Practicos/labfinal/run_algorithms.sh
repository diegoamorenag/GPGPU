#!/bin/bash

# Define the directory containing the images and the output base directory
image_dir="../imagenes"
output_dir="./evaluaciones"

# Array of window sizes to use
window_sizes=(3 5 7)  # Example window sizes

# List of images
images=("1.pgm" "Facultad_de_Ingenieria_UDELAR.pgm" "fing1_ruido.pgm")

# Loop through each image
for img in "${images[@]}"; do
    # Create an output directory specific to the image
    image_name=$(basename "$img" .pgm)  # Extract name without extension
    image_output_dir="$output_dir/$image_name"
    mkdir -p "$image_output_dir"  # Ensure the directory exists

    # Loop through each window size
    for w in "${window_sizes[@]}"; do
        # Define input and output file paths
        input_path="$image_dir/$img"
        output_path="$image_output_dir/salida_${w}.pgm"

        # Run the algorithm
        ./your_algorithm_binary "$input_path" "$output_path" "$w"
    done
done
