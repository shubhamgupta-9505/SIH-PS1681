'''
Created by Nikhat Singla on 19 November 2024
'''

from base64 import b64encode

# Import your encryption functions from their respective modules
from CryptographicAlgorithms.BlockCiphers.TripleDES import encrypt_DES3
from CryptographicAlgorithms.BlockCiphers.AES_128 import encrypt_AES_128
from CryptographicAlgorithms.BlockCiphers.AES_192 import encrypt_AES_192
from CryptographicAlgorithms.BlockCiphers.AES_256 import encrypt_AES_256
from CryptographicAlgorithms.BlockCiphers.Blowfish import encrypt_Blowfish
from CryptographicAlgorithms.BlockCiphers.CAST_128 import encrypt_CAST

from CryptographicAlgorithms.StreamCiphers.ARC4 import encrypt_ARC4
from CryptographicAlgorithms.StreamCiphers.ChaCha20 import encrypt_ChaCha20
from CryptographicAlgorithms.StreamCiphers.Salsa20 import encrypt_Salsa20

from CryptographicAlgorithms.HashFunctions.Blake import hash_BLAKE
from CryptographicAlgorithms.HashFunctions.Keccak import hash_KECCAK

# Read the large text file
with open('Datasets/Original/1MiB.txt', 'r') as file:
    data = file.read()

def extractFeatures(dictionary: dict) -> bytes:
    return f"{dictionary["algo_type"]},{dictionary["algo"]},{dictionary.get("mode")},".encode() + b64encode(dictionary.get("ciphertext", dictionary.get("digest"))) + b"\n"

# Divide the data into 1000 parts
part_size = len(data) // 1000
parts = [data[i:i+part_size].encode() for i in range(0, len(data), part_size)]

with open('Datasets/Encrypted/encrypted_base64.nsv', 'wb') as file:
    for part in parts:
        # Encrypt using Block Cipers (5 modes)
        for i in range(1, 6):
            file.write(extractFeatures(encrypt_AES_128(part, i)))
            file.write(extractFeatures(encrypt_AES_192(part, i)))
            file.write(extractFeatures(encrypt_AES_256(part, i)))
            file.write(extractFeatures(encrypt_DES3(part, i)))
            file.write(extractFeatures(encrypt_Blowfish(part, i)))
            file.write(extractFeatures(encrypt_CAST(part, i)))
        
        # Encrypt using Hash Functions
        file.write(extractFeatures(hash_BLAKE(part)))
        file.write(extractFeatures(hash_KECCAK(part)))

        # Encrypt using Stream Ciphers
        file.write(extractFeatures(encrypt_ARC4(part)))
        file.write(extractFeatures(encrypt_ChaCha20(part)))
        file.write(extractFeatures(encrypt_Salsa20(part)))

        # Encrypt using Public Key Ciphers

