mkdir -p results # Ensure the results directory exists
result_file="../results/all_results.txt" # Define the result file path

# Clear the result file at the start or create it if it doesn't exist
: > "$result_file"

cd a/ && make
cd ../ellos/ && make
cd ../a/ && ./biblios ../matrices/A_matrix.mtx >> "$result_file"
cd ../ellos/ && ./biblios ../matrices/A_matrix.mtx >> "$result_file"
cd ../a/ && ./biblios ../matrices/1.mtx >> "$result_file"
cd ../ellos/ && ./biblios ../matrices/1.mtx >> "$result_file"
cd ../a/ && ./biblios ../matrices/2.mtx >> "$result_file"
cd ../ellos/ && ./biblios ../matrices/2.mtx >> "$result_file"
cd ../a/ && ./biblios ../matrices/3.mtx >> "$result_file"
cd ../ellos/ && ./biblios ../matrices/3.mtx >> "$result_file"
cd ../a/ && ./biblios ../matrices/4.mtx >> "$result_file"
cd ../ellos/ && ./biblios ../matrices/4.mtx >> "$result_file"
cd ../a/ && ./biblios ../matrices/5.mtx >> "$result_file"
cd ../ellos/ && ./biblios ../matrices/5.mtx >> "$result_file"