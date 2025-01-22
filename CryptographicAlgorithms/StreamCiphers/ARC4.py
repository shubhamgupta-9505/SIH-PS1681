'''
Created by Nikhat Singla on 11 Nov 2024

This module implements ARC4 stream cipher
'''

from Crypto.Cipher import ARC4
from Crypto.Random import get_random_bytes

def encrypt_ARC4 (plaintext: bytes) -> dict:
    '''
    Description:
        This implementation uses 16 bytes key (recommended), although can vary between 1-256 bytes.

    Parameters:
        plaintext: a bytes representation of string

    Returns:
        Dictionary containing key-value pairs for all necessities
    '''
    

    key = get_random_bytes(16)

    cipher_encrypt = ARC4.new(key)

    ciphertext = cipher_encrypt.encrypt(plaintext)

    cipher_decrypt = ARC4.new(key)
    decrypted_text = cipher_decrypt.decrypt(ciphertext)

    return {"algo": "ARC4", "algo_type": "StreamCipher", "key": key.hex(), "decrypted_text": decrypted_text.hex(), "ciphertext": ciphertext}

# Example usage
if __name__ == "__main__":
    print(encrypt_ARC4(b"This is Sixteen. AndAllIsWell."))
