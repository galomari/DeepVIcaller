import numpy as np
import os
from scipy.io import loadmat
import pandas as pd
import matplotlib.pyplot as plt
import pysam

data_folder = os.path.join(os.path.dirname(__file__), '..', 'data')
output_folder = os.path.join(os.path.dirname(__file__), '..', 'VIcallerData')

if not os.path.exists(output_folder):
    os.makedirs(output_folder)

def read_fastq(filename):
    sequences = []
    with open(filename, 'r') as file:
        while True:
            file.readline()  # Skip the name line
            seq = file.readline().strip()  # Read the sequence line
            file.readline()  # Skip the + line
            file.readline()  # Skip the quality line
            if len(seq) == 0:
                break
            sequences.append(seq)
    return sequences

def read_bam(filename):
    sequences = []
    bamfile = pysam.AlignmentFile(filename, "rb")
    for read in bamfile.fetch(until_eof=True):
        if not read.is_unmapped:
            sequences.append(read.query_sequence)
    bamfile.close()
    return sequences

def seq_matrix(seq_list, dim):  # One Hot Encoding
    tensor = np.zeros((len(seq_list), dim, 4))
    for i in range(len(seq_list)):
        seq = seq_list[i]
        for j, s in enumerate(seq):
            if j >= dim:
                break
            if s == 'A' or s == 'a':
                tensor[i][j] = [1, 0, 0, 0]
            elif s == 'T' or s == 't':
                tensor[i][j] = [0, 1, 0, 0]
            elif s == 'C' or s == 'c':
                tensor[i][j] = [0, 0, 1, 0]
            elif s == 'G' or s == 'g':
                tensor[i][j] = [0, 0, 0, 1]
            else:
                tensor[i][j] = [0, 0, 0, 0]
    return tensor

def fastq_to_matrix(fastq_files, dim=2000):
    data_matrices = []
    for fastq_file in fastq_files:
        sequences = read_fastq(fastq_file)
        matrix = seq_matrix(sequences, dim)
        data_matrices.append(matrix)
    return np.concatenate(data_matrices, axis=0)

def bam_to_matrix(bam_files, dim=2000):
    data_matrices = []
    for bam_file in bam_files:
        sequences = read_bam(bam_file)
        matrix = seq_matrix(sequences, dim)
        data_matrices.append(matrix)
    return np.concatenate(data_matrices, axis=0)

# Example usage
fastq_files = [
    os.path.join(data_folder, 'test2.fastq')
    # Add paths to your FASTQ files
]

bam_files = [
    os.path.join(data_folder, 'seq_RNA.bam')
    # Add paths to your BAM files
]

# Convert FASTQ data to one-hot encoded matrices
fastq_data_matrix = fastq_to_matrix(fastq_files, dim=2000)
# Convert BAM data to one-hot encoded matrices
bam_data_matrix = bam_to_matrix(bam_files, dim=2000)

# Save the matrices
np.save(os.path.join(output_folder, 'vicaller_fastq.npy'), fastq_data_matrix)
np.save(os.path.join(output_folder, 'vicaller_bam.npy'), bam_data_matrix)

# Generate sample labels for demonstration (replace with actual labels)
num_samples_fastq = fastq_data_matrix.shape[0]
num_samples_bam = bam_data_matrix.shape[0]
labels_fastq = np.random.randint(2, size=num_samples_fastq)
labels_bam = np.random.randint(2, size=num_samples_bam)

# Save the data and labels
np.save(os.path.join(output_folder, 'fatq_Test_Data.npy'), fastq_data_matrix)
np.save(os.path.join(output_folder, 'fastqlable_Test_Label.npy'), labels_fastq)
np.save(os.path.join(output_folder, 'bam_Test_Data.npy'), bam_data_matrix)
np.save(os.path.join(output_folder, 'bam_label_Test_Label.npy'), labels_bam)

def plot_one_hot_encoding(matrix, num_samples=5):
    fig, axes = plt.subplots(num_samples, 1, figsize=(10, 2 * num_samples))
    for i in range(num_samples):
        axes[i].imshow(matrix[i].T, aspect='auto', cmap='viridis')
        axes[i].set_title(f'Sample {i + 1}')
        axes[i].set_xlabel('Position in Sequence')
        axes[i].set_ylabel('Nucleotide')
        axes[i].set_yticks([0, 1, 2, 3])
        axes[i].set_yticklabels(['A', 'T', 'C', 'G'])
    plt.tight_layout()
    plt.show()

def display_data_matrix_head(data_matrix, num_samples=5, num_positions=10):
    for i in range(num_samples):
        print(f"Sample {i + 1}:")
        print(data_matrix[i, :num_positions, :])
        print("\n")

# Displaying the head of the data_matrix
print("FASTQ Data Matrix Head:")
display_data_matrix_head(fastq_data_matrix)

print("BAM Data Matrix Head:")
display_data_matrix_head(bam_data_matrix)

# Check the shape of the data matrix
print("Shape of the FASTQ data matrix:", fastq_data_matrix.shape)
print("Shape of the BAM data matrix:", bam_data_matrix.shape)

# Display the one-hot encoded matrices for a few samples
print("FASTQ Data Matrix Visualization:")
plot_one_hot_encoding(fastq_data_matrix)

print("BAM Data Matrix Visualization:")
plot_one_hot_encoding(bam_data_matrix)

