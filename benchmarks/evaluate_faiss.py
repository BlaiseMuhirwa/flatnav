import numpy as np
import faiss
from keras.datasets import mnist


"""
some parameters (similar to what Fiass' HNSW uses): 
    - number of links per node: 32 
    - 
"""



def load_data():
    # Load MNIST dataset
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    
    # Reshape and normalize
    x_train = x_train.reshape(-1, 784).astype(np.float32)
    x_test = x_test.reshape(-1, 784).astype(np.float32)
    x_train /= 255.
    x_test /= 255.

    return x_train, y_train, x_test, y_test

def train_hnsw_index(data, serialize=True):
    # configure the index 
    d = data.shape[1]  # data dimension
    m = 8  # number of subquantizers
    nbits = 8  # bits per subvector index
    
    # Create the HNSW quantizer
    quantizer = faiss.IndexHNSWFlat(d, 16)  # 16 is a typical value for HNSW M parameter
    quantizer.hnsw.efConstruction = 128  # Typical values are between 20-64
    quantizer.hnsw.efSearch = 128  # Typical values are between 20-64

    # Create the IVFPQ index
    index = faiss.IndexIVFPQ(quantizer, d, 100, m, nbits)

    # Train the index
    print("Training index...")
    index.train(data)

    # Add vectors to the index
    index.add(data)
    
    if serialize:
        # Serialize the index to disk
        buffer = faiss.serialize_index(index)
        index_size = len(buffer)
        
        print(f"Index size: {index_size} bytes")
    
    return index


def compute_recall(index, x_test, y_train, y_test, k=100):
    D, I = index.search(x_test, k)
    correct = 0

    # For each test point, check if its true label is among the k-nearest neighbors
    for i, neighbors in enumerate(I):
        labels = [y_train[idx] for idx in neighbors]
        if y_test[i] in labels:
            correct += 1

    recall = correct / len(y_test)
    return recall

def main():
    x_train, y_train, x_test, y_test = load_data()
    index = train_hnsw_index(x_train)
    recall = compute_recall(index, x_test, y_train, y_test)
    print(f"Recall@100: {recall}")

if __name__ == "__main__":
    main()
