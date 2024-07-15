#!/bin/bash

# Define the directory containing the images and the output base directory
image_dir="./imagenes"
output_dir="./evaluaciones"

# Array of window sizes to use
window_sizes=(3 5 7 9 11)  # Example window sizes

# List of images
images=("1.pgm" "Facultad_de_Ingenieria_UDELAR.pgm" "fing1_ruido.pgm")

# Define executable directories
exec_dirs=( "CPU/median_filter" "GPU/bancos/bancos" "GPU/baseline/baseline" "GPU/counting/counting" "GPU/memoria_compartida/memoria_compartida" \
            "GPU/network/network" "GPU/quickselect/quickselect" "GPU/radix_filter/radix_filter" "GPU/radixCub/radixCub" \
            "GPU/radixCurso/radixCurso" "GPU/shell/shell" "GPU/texturas/texturas" "GPU/thrust_filter/thrust_filter" )

# Loop through each executable directory
for exec_dir in "${exec_dirs[@]}"; do
    # Get executable name
    exec_name=$(basename "$exec_dir")

    # Loop through each image
    for img in "${images[@]}"; do
        # Create an output directory specific to the image and algorithm
        image_name=$(basename "$img" .pgm)  # Extract name without extension
        image_output_dir="$output_dir/$image_name/$exec_name"
        mkdir -p "$image_output_dir"  # Ensure the directory exists

        # Loop through each window size
        for w in "${window_sizes[@]}"; do
            # Define input and output file paths
            input_path="$image_dir/$img"
            output_path="$image_output_dir/salida_${w}.pgm"

            # Run the algorithm
            ./"$exec_dir" "$input_path" "$output_path" "$w"
        done
    done
done