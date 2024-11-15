'''
Created by Nikhat Singla on 15 November 2024
'''

import os
import importlib

# Import your encryption functions from their respective modules
from CryptographicAlgorithms.BlockCiphers.TripleDES import encrypt_DES3
from CryptographicAlgorithms.BlockCiphers.AES_128 import encrypt_AES
from CryptographicAlgorithms.BlockCiphers.Blowfish import encrypt_Blowfish
from CryptographicAlgorithms.BlockCiphers.CAST_128 import encrypt_CAST

from CryptographicAlgorithms.StreamCiphers.ARC4 import encrypt_ARC4
from CryptographicAlgorithms.StreamCiphers.ChaCha20 import encrypt_ChaCha20
from CryptographicAlgorithms.StreamCiphers.Salsa20 import encrypt_Salsa20

from CryptographicAlgorithms.HashFunctions.Blake import encrypt_BLAKE
from CryptographicAlgorithms.HashFunctions.Keccak import encrypt_ARC4



rsa_module = importlib.import_module("Cryptographic Algorithms.BlockCiphers.AES_128")

def read_large_file(file_path):
    """Read a large file and yield chunks of data."""
    with open(file_path, 'r', encoding='utf-8') as file:
        while True:
            data = file.read(1024 * 1024)  # Read in 1 MB chunks
            if not data:
                break
            yield data

def divide_into_parts(data, num_parts):
    """Divide the data into approximately equal parts."""
    part_size = len(data) // num_parts
    return [data[i * part_size:(i + 1) * part_size] for i in range(num_parts)]

def encrypt_parts(parts):
    """Encrypt each part using all available algorithms."""
    encrypted_parts = []
    for part in parts:
        encrypted_rsa = encrypt_RSA(part)
        encrypted_aes = encrypt_AES(part)
        encrypted_parts.append((encrypted_rsa, encrypted_aes))
    return encrypted_parts

def write_encrypted_parts(encrypted_parts, output_file):
    """Write the encrypted parts to an output file."""
    with open(output_file, 'w', encoding='utf-8') as file:
        for rsa_part, aes_part in encrypted_parts:
            file.write(f"RSA: {rsa_part}\n")
            file.write(f"AES: {aes_part}\n")

def main():
    input_file = 'large_plaintext.txt'  # Path to your large plaintext file
    output_file = 'encrypted_output.txt'  # Path to save the encrypted output
    num_parts = 1000  # Number of parts to divide the plaintext into

    # Read the large file and concatenate all parts into a single string
    full_data = ''.join(read_large_file(input_file))

    # Divide the full data into parts
    parts = divide_into_parts(full_data, num_parts)

    # Encrypt the parts
    encrypted_parts = encrypt_parts(parts)

    # Write the encrypted parts to the output file
    write_encrypted_parts(encrypted_parts, output_file)

if __name__ == '__main__':
    main()
