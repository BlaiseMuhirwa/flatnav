import scipy.io
import matplotlib.pyplot as plt
import os 
import numpy as np 


def read_matrix_market(filename):
    with open(filename, 'r') as f:
        # Skip header
        f.readline()
        
        # Read matrix dimensions
        rows, cols, _ = list(map(int, f.readline().split()))

        # Initialize edge counts
        edge_counts = np.zeros(rows, dtype=int)

        # Read and process each line
        for line in f:
            row, col = list(map(int, line.split()))
            edge_counts[row - 1] += 1  # Convert to 0-based index

        return edge_counts

# Read the file and get edge counts
file_path = file_path = os.path.join(os.getcwd(), "data", "mnist-784-euclidean", 'mnist2.mtx')
edge_counts = read_matrix_market(file_path)

# Plot the histogram
plt.hist(edge_counts, bins=30, edgecolor='black')
plt.title('Distribution of Number of Edges per Node')
plt.xlabel('Number of Edges')
plt.ylabel('Frequency')
plt.show()

