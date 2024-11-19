'''
Created by Nikhat Singla on 19 November 2024
'''

import pickle
from concurrent.futures import ProcessPoolExecutor, as_completed

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

# Set the chunk size to 1MB
chunk_size = 1024 * 1024  # 1MB

# Open the output file for writing
with open('encrypted.txt', 'wb') as output_file:
    with open('full.txt', 'rb') as input_file:
        while True:
            chunk = input_file.read(chunk_size)
            if not chunk:
                break

            def encrypt_chunk(chunk):
                all_enc = []

                # Encrypt using Block Ciphers (5 modes)
                for i in range(1, 6):
                    all_enc.append(encrypt_AES(chunk, i))
                    all_enc.append(encrypt_DES3(chunk, i))
                    all_enc.append(encrypt_Blowfish(chunk, i))
                    all_enc.append(encrypt_CAST(chunk, i))

                # Encrypt using Hash Functions
                all_enc.append(hash_BLAKE(chunk))
                all_enc.append(hash_KECCAK(chunk))

                # Encrypt using Stream Ciphers
                all_enc.append(encrypt_ARC4(chunk))
                all_enc.append(encrypt_ChaCha20(chunk))
                all_enc.append(encrypt_Salsa20(chunk))

                return all_enc

            with ProcessPoolExecutor() as executor:
                futures = [executor.submit(encrypt_chunk, chunk)]
                for encrypted_part in [future.result() for future in as_completed(futures)]:
                    pickle.dump(encrypted_part, output_file)
