'''
Created by Nikhat Singla on 11 Nov 2024

This module implements Salsa20 stream cipher
'''

from Crypto.Cipher import Salsa20
from Crypto.Random import get_random_bytes

def encrypt_Salsa20 (plaintext: bytes) -> dict:
    '''
    Description:
        This implementation encrypts plaintext using a random 32-byte key and an 8-byte nonce, although 16-byte key can also be used.

    Parameters:
        plaintext: a bytes representation of string

    Returns:
        Dictionary containing key-value pairs for all necessities
    '''

    key = get_random_bytes(32)

    cipher_encrypt = Salsa20.new(key=key)
    ciphertext = cipher_encrypt.encrypt(plaintext)
    
    nonce = cipher_encrypt.nonce

    cipher_decrypt = Salsa20.new(key=key, nonce=nonce)
    decrypted_text = cipher_decrypt.decrypt(ciphertext)

    return {"algo": "Salsa20", "algo_type": "StreamCipher", "key": key.hex(), "nonce": nonce.hex(), "decrypted_text": decrypted_text.hex(), "ciphertext": ciphertext}

# Example usage
if __name__ == "__main__":
    print(encrypt_Salsa20(b"This is Sixteen. AndAllIsWell."))