import pandas as pd
from joblib import Parallel, delayed

import pandas as pd
from joblib import Parallel, delayed
import math
from itertools import groupby

import numpy as np

def transformer_str(hex_string):
    # Split hex string into pairs of two characters
    pairs = [hex_string[i:i+2] for i in range(0, len(hex_string), 2)]

    # Convert to integers and then binary strings with padding
    int_array = np.array([int(pair, 16) for pair in pairs], dtype=np.uint8)
    binary_array = np.vectorize(np.binary_repr)(int_array, width=8)

    # Concatenate all binary strings
    return ''.join(binary_array)

def transform_row(dataset):
    """
    Append features to the dataset for each ciphertext in column 4 using parallelization.

    Parameters:
        dataset (pd.DataFrame): Original dataset containing binary ciphertexts.
        segment_sizes (list): List of segment sizes to extract features for.
        n_jobs (int): Number of parallel jobs (-1 means using all available processors).

    Returns:
        pd.DataFrame: Updated dataset with appended feature columns.
    """
    # Create a list of tasks to process in parallel
    tasks = [delayed(transformer_str)(row[3]) for _, row in dataset.iterrows()]

    # Execute tasks in parallel
    results = Parallel(n_jobs=-1)(tasks)

    return results

# pip install Pywavelets

# pip install pyunicorn

# pip install nolds

def entropy_analysis(k):
  #this function will be used to measure the randomness of cipher text this feature
  # might be a little less useful
  prob_0 = k.count('0') / len(k)
  prob_1 = k.count('1') / len(k)

    # Avoid log(0) by ensuring probabilities > 0
  entropy = 0
  if prob_0 > 0:
      entropy -= prob_0 * np.log2(prob_0)
  if prob_1 > 0:
      entropy -= prob_1 * np.log2(prob_1)

  return entropy

def frequency_test(k):
  #checks the deviation form stability
    E_i = 0.5
    prob_0 = k.count('0') / len(k)
    prob_1 = k.count('1') / len(k)
    chi_squared = (((prob_0 - E_i) ** 2) / E_i) + (((prob_1 - E_i) ** 2) / E_i)
    deviation = chi_squared ** 0.5  # Taking the square root for stability test

    return deviation

def run_length(k):
  #used to find the A run of length k
  #consists of exactly k identical bits and
  #is bounded before and after with a bit of opposite value.
  runs = [(key, len(list(group))) for key, group in groupby(k)]

    # Separate runs of 0s and 1s
  run_lengths_0 = [length for bit, length in runs if bit == '0']
  run_lengths_1 = [length for bit, length in runs if bit == '1']

    # Calculate mean run lengths, handle empty cases gracefully
  mean_run_0 = np.mean(run_lengths_0) if run_lengths_0 else 0
  mean_run_1 = np.mean(run_lengths_1) if run_lengths_1 else 0

  return [mean_run_0, mean_run_1]

def hamming_weight(k):
  #counts number of 1
  return k.count('1')

def bit_transition(segment):
    """
    Count the number of bit transitions (changes from 0 to 1 or 1 to 0) in the binary string.

    Parameters:
        segment (str): Binary string.

    Returns:
        int: Number of transitions.
    """
    # Convert the binary string to a list of integers
    data = list(map(int, segment))

    # Initialize count of transitions
    count = 0

    # Loop through adjacent bits
    for i in range(len(data) - 1):
        count += abs(data[i] - data[i + 1])  # Compute transition

    return count

def binary_skewness_kurtosis(k):
    # Convert binary string into a list of integers
    data = list(map(int, k))
    n = len(data)
    # Calculate Mean
    mean = sum(data) / n
    # Calculate Median
    sorted_data = sorted(data)
    if n % 2 == 1:  # Odd number of elements
        median = sorted_data[n // 2]
    else:  # Even number of elements
        median = (sorted_data[n // 2 - 1] + sorted_data[n // 2]) / 2
    # Calculate Standard Deviation
    variance = sum((x - mean) ** 2 for x in data) / n
    std_dev = variance ** 0.5
    if std_dev == 0:  # Handle edge case where all elements are identical
        return 0,0
    # Calculate Skewness
    skewness = 3 * (mean - median) / std_dev
    # Calculate Kurtosis
    kurtosis = sum((x - mean) ** 4 for x in data) / (n * std_dev ** 4) - 3

    return skewness, kurtosis


def autocorrelation(binary_seq, lag):
    # Convert binary string into a list of integers
    data = list(map(int, binary_seq))
    n = len(data)
    if lag >= n:
        return "Lag is too large for the sequence length."
    # Calculate Mean
    mean = sum(data) / n
    # Calculate Variance
    variance = sum((x - mean) ** 2 for x in data) / n
    if variance == 0:  # Handle edge case where all elements are identical
        return 0
    # Calculate Autocorrelation for the given lag
    autocorr = sum((data[t] - mean) * (data[t + lag] - mean) for t in range(n - lag)) / variance

    return autocorr

def fractal_dimension(binary_seq):
    """
    Compute the fractal dimension of a binary sequence.

    Parameters:
        binary_seq (str): Binary string representing the sequence.

    Returns:
        float: Fractal dimension of the sequence.
    """
    # Convert binary string into a list of integers
    data = list(map(int, binary_seq))
    n = len(data)

    # Define possible scales
    scales = [2**i for i in range(1, int(math.log2(n)) + 1)] if n > 1 else []

    log_r = []
    log_Nr = []

    for r in scales:
        # Divide sequence into chunks of size r
        chunks = [data[i:i + r] for i in range(0, n, r)]

        # Count non-empty chunks (chunks with at least one "1")
        N_r = sum(1 for chunk in chunks if sum(chunk) > 0)

        # Avoid log(0) errors
        if N_r > 0 and r > 0:
            log_r.append(math.log(r))
            log_Nr.append(math.log(N_r))

    # Handle edge cases where log lists are empty
    if len(log_r) < 2 or len(log_Nr) < 2:
        return 0.0  # Cannot compute fractal dimension for insufficient data

    # Calculate slope (Fractal Dimension) using the formula: slope = Δy / Δx
    D = (log_Nr[-1] - log_Nr[0]) / (log_r[-1] - log_r[0])
    return D

def extract_features(binary_string):
    """
    Extract features from binary ciphertext for training a model.

    Parameters:
        binary_string (str): Binary sequence representing the ciphertext.

    Returns:
        dict: A dictionary containing extracted features.
    """
    import numpy as np

    # Feature: Length of ciphertext
    length = len(binary_string)

    # Feature: Frequency of '1's and '0's
    ones = binary_string.count('1') / length
    zeros = binary_string.count('0') / length

    # Feature: Spectral Density (FFT mean and variance)
    numeric_ciphertext = np.array([int(bit) for bit in binary_string]) * 2 - 1
    fft_result = np.fft.fft(numeric_ciphertext)
    spectral_density = np.abs(fft_result) ** 2
    spectral_mean = np.mean(spectral_density)
    spectral_variance = np.var(spectral_density)

    # Compile features into a dictionary
    features = [
         spectral_mean,
         spectral_variance
    ]

    return features

def compute_convolution(ciphertext, kernel=[1,-1]):
    """
    Compute the convolution of a ciphertext with a given kernel.

    Parameters:
        ciphertext (list or np.array): Numeric representation of the ciphertext.
        kernel (list or np.array): Convolutional filter or kernel.

    Returns:
        np.array: Convolution result.
    """
    # Convert ciphertext and kernel to numpy arrays for efficient computation
    ciphertext = np.array(ciphertext)
    kernel = np.array(kernel)
    # Lengths of ciphertext and kernel
    ciphertext_len = len(ciphertext)
    kernel_len = len(kernel)
    # Length of the output
    output_len = ciphertext_len - kernel_len + 1
    # Initialize the convolution result
    convolution_result = np.zeros(output_len)
    # Perform the convolution
    for i in range(output_len):
        convolution_result[i] = np.sum(ciphertext[i:i+kernel_len] * kernel)
    return convolution_result

import zlib

def compute_ciphertext_complexity(ciphertext):
    """
    Compute the ciphertext complexity using zlib compression.

    Parameters:
        ciphertext (str): Ciphertext as a string.

    Returns:
        int: Length of the compressed ciphertext.
    """
    # Compress the ciphertext using zlib
    compressed_ciphertext = zlib.compress(ciphertext.encode())

    # Return the length of the compressed ciphertext
    return len(compressed_ciphertext)

import pandas as pd
from joblib import Parallel, delayed

def extract_features_for_row(binary_string, segment_sizes):
    """
    Extract features for a single row of the dataset.

    Parameters:
        row (pd.Series): A row from the dataset containing binary ciphertext in column 4.
        segment_sizes (list): List of segment sizes to extract features for.

    Returns:
        dict: A dictionary of features extracted for the row.
    """
    features = {}
   
    # Extract features for each segment size
    for size in segment_sizes:
        # Extract the segment of the required size
        segment = binary_string[:size]  # Binary segment (first `size` bits)

        # Pad with zeros if the segment is smaller than the required size
        segment = segment.ljust(size, '0')

        # Extract features for the segment
        features[f"entropy_{size}"] = entropy_analysis(segment)
        features[f"frequency_deviation_{size}"] = frequency_test(segment)
        features[f"mean_run_0_{size}"], features[f"mean_run_1_{size}"] = run_length(segment)
        features[f"hamming_weight_{size}"] = hamming_weight(segment)
        features[f"bit_transitions_{size}"] = bit_transition(segment)
        features[f"skewness_{size}"], features[f"kurtosis_{size}"] = binary_skewness_kurtosis(segment)
        features[f"autocorrelation_{size}"] = autocorrelation(segment, lag=4)
        features[f"fractal_dimension_{size}"] = fractal_dimension(segment)
        spectral_features = extract_features(segment)
        features[f"spectral_mean_{size}"] = spectral_features[0]
        features[f"spectral_variance_{size}"] = spectral_features[1]
        convolution_result = compute_convolution([int(bit) for bit in segment])
        features[f"convolution_{size}"] = sum(convolution_result)
        features[f"complexity_{size}"] = compute_ciphertext_complexity(segment)

    return features

def append_features_to_dataset(binary_string, segment_sizes=[128,256,512,1024,2048,4096,8192], n_jobs=-1):
    """
    Append features to the dataset for each ciphertext in column 4 using parallelization.

    Parameters:
        dataset (pd.DataFrame): Original dataset containing binary ciphertexts.
        segment_sizes (list): List of segment sizes to extract features for.
        n_jobs (int): Number of parallel jobs (-1 means using all available processors).

    Returns:
        pd.DataFrame: Updated dataset with appended feature columns.
    """
    # Create a list of tasks to process in parallel
    results = extract_features_for_row(binary_string, segment_sizes)

    return results


