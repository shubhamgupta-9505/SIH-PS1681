import pandas as pd
from itertools import groupby
import math
def transformer_str(k):
    binary_string_final = ""
    for hex_string in k:
        pairs = [hex_string[i:i+2] for i in range(0, len(hex_string), 2)]
        binary_list = [bin(int(pair, 16))[2:].zfill(8) for pair in pairs]
        binary_string = ''.join(binary_list)
        binary_string_final += binary_string
    return binary_string_final

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

def chi_square_test(segment):
    """
    Perform the chi-square test on the given segment.
    
    Parameters:
        segment (str): The binary string segment to test.

    Returns:
        float: The chi-square test result.
    """
    # Convert the binary string into bytes (encode as ASCII)
    data = bytes(segment, 'ascii')  # Use 'ascii' encoding for binary strings
    
    if len(data) == 0:
        return 0
    
    observed = np.bincount(np.frombuffer(data, dtype=np.uint8), minlength=256)
    expected = np.full(256, len(data) / 256)
    chi_square = np.sum((observed - expected) ** 2 / expected)
    
    return chi_square


import numpy as np
import pywt
import matplotlib.pyplot as plt

def wavelet_transform_features(data, wavelet='haar', max_level=5):
    """
    Perform wavelet transform and extract features from the coefficients.

    Parameters:
        data (str): Binary ciphertext string.
        wavelet (str): Type of wavelet to use for the transform.
        max_level (int): Maximum level of decomposition.

    Returns:
        dict: Extracted wavelet features.
    """
    if len(data) == 0:
        return {}

    # Convert binary string to numeric values (1 for '1', -1 for '0')
    numeric_data = np.array([1 if bit == '1' else -1 for bit in data])

    # Perform wavelet transform
    coeffs = pywt.wavedec(numeric_data, wavelet, level=max_level)

    # Extract features from each level
    features = {}
    for level, coeff in enumerate(coeffs):
        features[f'wavelet_mean_level_{level}'] = np.mean(coeff)
        features[f'wavelet_std_level_{level}'] = np.std(coeff)
        features[f'wavelet_energy_level_{level}'] = np.sum(np.square(coeff))
        features[f'wavelet_entropy_level_{level}'] = -np.sum(coeff * np.log2(np.abs(coeff + 1e-10)))

    return features

from scipy.fft import fft
from scipy.stats import entropy

def spectral_entropy(binary_string):
    numeric_data = np.array([1 if bit == '1' else -1 for bit in binary_string])
    fft_values = np.abs(fft(numeric_data))
    normalized_fft = fft_values / np.sum(fft_values)
    return entropy(normalized_fft)

def dominant_frequency_magnitude(binary_string):
    numeric_data = np.array([1 if bit == '1' else -1 for bit in binary_string])
    fft_values = np.abs(fft(numeric_data))
    return np.max(fft_values)

def power_spectral_density_peaks(binary_string, num_peaks=3):
    numeric_data = np.array([1 if bit == '1' else -1 for bit in binary_string])
    fft_values = np.abs(fft(numeric_data)) ** 2
    return sorted(fft_values, reverse=True)[:num_peaks]

def entropy_gradient(binary_string, block_size):
    entropies = []
    for i in range(0, len(binary_string), block_size):
        block = binary_string[i:i + block_size]
        counts = np.bincount([int(bit) for bit in block], minlength=2)
        probabilities = counts / counts.sum()
        entropies.append(-np.sum(probabilities * np.log2(probabilities + 1e-10)))
    return np.mean(np.gradient(entropies))

def adjacent_bit_probability(binary_string):
    transitions = [1 if binary_string[i] != binary_string[i + 1] else 0 for i in range(len(binary_string) - 1)]
    return sum(transitions) / len(transitions)

def bit_pair_frequency(binary_string):
    pairs = [binary_string[i:i + 2] for i in range(len(binary_string) - 1)]
    return {pair: pairs.count(pair) / len(pairs) for pair in ['00', '01', '10', '11']}

import nolds

def hurst_exponent(binary_string):
    numeric_data = np.array([1 if bit == '1' else -1 for bit in binary_string])
    return nolds.hurst_rs(numeric_data)

def detrended_fluctuation_analysis(binary_string):
    numeric_data = np.array([1 if bit == '1' else -1 for bit in binary_string])
    return nolds.dfa(numeric_data)

from pyunicorn.timeseries.recurrence_plot import RecurrencePlot

def recurrence_quantification_analysis(binary_string, threshold=0.5):
    numeric_data = np.array([1 if bit == '1' else -1 for bit in binary_string])
    rp = RecurrencePlot(numeric_data, threshold=threshold)
    return rp.recurrence_rate()

def wavelet_transform_features(data, wavelet='haar', max_level=5):
    """
    Perform wavelet transform and extract features from the coefficients.

    Parameters:
        data (str): Binary ciphertext string.
        wavelet (str): Type of wavelet to use for the transform.
        max_level (int): Maximum level of decomposition.

    Returns:
        dict: Extracted wavelet features.
    """
    if len(data) == 0:
        return {}

    # Convert binary string to numeric values (1 for '1', -1 for '0')
    numeric_data = np.array([1 if bit == '1' else -1 for bit in data])

    # Perform wavelet transform
    coeffs = pywt.wavedec(numeric_data, wavelet, level=max_level)

    # Extract features from each level
    features = {}
    for level, coeff in enumerate(coeffs):
        features[f'wavelet_mean_level_{level}'] = np.mean(coeff)
        features[f'wavelet_std_level_{level}'] = np.std(coeff)
        features[f'wavelet_energy_level_{level}'] = np.sum(np.square(coeff))
        features[f'wavelet_entropy_level_{level}'] = -np.sum(coeff * np.log2(np.abs(coeff + 1e-10)))

    return features

import pandas as pd
from joblib import Parallel, delayed

def extract_features_for_row(row, segment_sizes):
    """
    Extract features for a single row of the dataset.

    Parameters:
        row (pd.Series): A row from the dataset containing binary ciphertext in column 4.
        segment_sizes (list): List of segment sizes to extract features for.

    Returns:
        dict: A dictionary of features extracted for the row.
    """
    features = {}  # Initialize as a dictionary
    binary_string = row[4]  # Column containing binary ciphertext

    # Extract features for each segment size
    for size in segment_sizes:
        # Extract the segment of the required size
        segment = binary_string[:size]  # Binary segment (first size bits)

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
        features[f"chi_square_{size}"] = chi_square_test(segment)  # Updated call
        wavelet_features = wavelet_transform_features(segment)
        for key, value in wavelet_features.items():
            features[f"{key}_{size}"] = value
        features[f"recurrence_rate_{size}"] = recurrence_quantification_analysis(segment)
        features[f"hurst_exponent_{size}"] = hurst_exponent(segment)
        features[f"dfa_{size}"] = detrended_fluctuation_analysis(segment)
        bit_freq = bit_pair_frequency(segment)
        for pair, freq in bit_freq.items():
            features[f"bit_pair_freq_{pair}_{size}"] = freq
        features[f"entropy_gradient_{size}"] = entropy_gradient(segment,4)
        features[f"adjacent_bit_probability_{size}"] = adjacent_bit_probability(segment)
        features[f"spectral_entropy_{size}"] = spectral_entropy(segment)
        features[f"dominant_frequency_magnitude_{size}"] = dominant_frequency_magnitude(segment)
        peaks = power_spectral_density_peaks(segment)
        for i, peak in enumerate(peaks):
            features[f"power_spectral_peak_{i+1}_{size}"] = peak

    return features



def append_features_to_dataset(dataset, segment_sizes=[128,256,512,1024,2048,4096], n_jobs=-1):
    """
    Append features to the dataset for each ciphertext in column 4 using parallelization.

    Parameters:
        dataset (pd.DataFrame): Original dataset containing binary ciphertexts.
        segment_sizes (list): List of segment sizes to extract features for.
        n_jobs (int): Number of parallel jobs (-1 means using all available processors).

    Returns:
        pd.DataFrame: Updated dataset with appended feature columns.
    """
    # Extract features in parallel for each row
    results = Parallel(n_jobs=n_jobs)(
        delayed(extract_features_for_row)(row, segment_sizes) for _, row in dataset.iterrows()
    )

    # Combine the features back into the DataFrame
    features_df = pd.DataFrame(results)
    updated_dataset = pd.concat([dataset.reset_index(drop=True), features_df], axis=1)

    return updated_dataset

df=pd.read_csv("encrypted copy.csv",header=None,nrows=50000)
new_df=df[df[0]!='HashFun'].copy()
new_df[4]=new_df[3].apply(transformer_str)
updated_dataset=append_features_to_dataset(new_df)
updated_dataset.to_csv("store.csv")