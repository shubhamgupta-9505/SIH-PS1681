'''
Created by Nikhat Singla on 15 November 2024
'''

import os
import multiprocessing as mp
from functools import partial
import json
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

data = data.encode()
# print(data)

# Divide the data into 1000 parts
part_size = len(data) // 1000
parts = [data[i:i+part_size] for i in range(0, len(data), part_size)]

# print(parts)

# Encrypt each part with 15 different cryptographic algorithms
encrypted_parts = []

for part in parts:
    all_enc = []

    for i in range(1, 6):
        all_enc.append(encrypt_AES(part, i))
        all_enc.append(encrypt_DES3(part, i))

    encrypted_parts.append(all_enc)


# print(encrypted_parts)
# Store the encrypted parts in one file
with open('/home/nikhatsingla/Documents/SIH-PS1681/sampleoutput.txt', 'wb') as file:
    # json.dump(str(dataset), file, indent=4)
    pickle.dump(encrypted_parts, file)

with open('/home/nikhatsingla/Documents/SIH-PS1681/sampleoutput.txt', 'rb') as file:
    myfile = pickle.load(file)

print(myfile)