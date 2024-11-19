'''
Created by Nikhat Singla on 10 Nov 2024

This module implements ChaCha20 stream cipher
'''

from Crypto.Cipher import ChaCha20
from Crypto.Random import get_random_bytes

def encrypt_ChaCha20 (plaintext: bytes) -> dict:
    '''
    Description:
        This implementation encrypts plaintext using a random 32-byte key and an 8-byte nonce, although it can also accommodate a 12-byte nonce.

    Parameters:
        plaintext: a bytes representation of string

    Returns:
        Dictionary containing key-value pairs for all necessities
    '''

    key = get_random_bytes(32)

    cipher_encrypt = ChaCha20.new(key=key)
    nonce = cipher_encrypt.nonce

    ciphertext = cipher_encrypt.encrypt(plaintext)

    cipher_decrypt = ChaCha20.new(key=key, nonce=nonce)
    decrypted_text = cipher_decrypt.decrypt(ciphertext)

    return {"algo": "ChaCha20", "algo_type": "StreamCipher", "key": key.hex(), "nonce": nonce.hex(), "decrypted_text": decrypted_text.decode(), "ciphertext": ciphertext.hex()}

# Example usage
if __name__ == "__main__":
    print(encrypt_ChaCha20(b"This is Sixteen. AndAllIsWell."))