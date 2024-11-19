'''
Created by Nikhat Singla on 19 November 2024
'''

import pickle

# Import your encryption functions from their respective modules
from CryptographicAlgorithms.BlockCiphers.TripleDES import encrypt_DES3
from CryptographicAlgorithms.BlockCiphers.AES_128 import encrypt_AES
from CryptographicAlgorithms.BlockCiphers.Blowfish import encrypt_Blowfish
from CryptographicAlgorithms.BlockCiphers.CAST_128 import encrypt_CAST

from CryptographicAlgorithms.StreamCiphers.ARC4 import encrypt_ARC4
from CryptographicAlgorithms.StreamCiphers.ChaCha20 import encrypt_ChaCha20
from CryptographicAlgorithms.StreamCiphers.Salsa20 import encrypt_Salsa20

from CryptographicAlgorithms.HashFunctions.Blake import hash_BLAKE
from CryptographicAlgorithms.HashFunctions.Keccak import hash_KECCAK

# Read the large text file
with open('/home/nikhatsingla/Documents/SIH-PS1681/testsample.txt', 'r') as file:
    data = file.read()

# Convert data into bytes format
data = data.encode()

# Divide the data into 25000 parts
part_size = len(data) // 25000
parts = [data[i:i+part_size] for i in range(0, len(data), part_size)]

# Encrypt each part with all cryptographic algorithms
encrypted_parts = []

for part in parts:
    all_enc = []

    # Encrypt using Block Cipers (5 modes)
    for i in range(1, 6):
        all_enc.append(encrypt_AES(part, i))
        all_enc.append(encrypt_DES3(part, i))
        all_enc.append(encrypt_Blowfish(part, i))
        all_enc.append(encrypt_CAST(part, i))
    
    # Encrypt using Hash Functions
    all_enc.append(hash_BLAKE(part))
    all_enc.append(hash_KECCAK(part))

    # Encrypt using Stream Ciphers
    all_enc.append(encrypt_ARC4(part))
    all_enc.append(encrypt_ChaCha20(part))
    all_enc.append(encrypt_Salsa20(part))

    # Encrypt using Public Key Ciphers


    # Append to dataset
    encrypted_parts.append(all_enc)

with open('/home/nikhatsingla/Documents/SIH-PS1681/outputsample.txt') as file:
    pickle(encrypted_parts, file)