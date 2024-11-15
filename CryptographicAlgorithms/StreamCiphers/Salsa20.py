'''
Created by Nikhat Singla on 11 Nov 2024
'''

def encrypt_Salsa20 (plaintext: bytes):
    '''
    Parameter "plaintext" is of type "bytes". 32-byte key and 8-byte nonce (default) is used.
    '''

    from Crypto.Cipher import Salsa20
    from Crypto.Random import get_random_bytes
    from base64 import b64encode

    key = get_random_bytes(32)

    cipher_encrypt = Salsa20.new(key=key)
    ciphertext = cipher_encrypt.encrypt(plaintext)
    nonce = cipher_encrypt.nonce

    cipher_decrypt = Salsa20.new(key=key, nonce=nonce)
    decrypted_text = cipher_decrypt.decrypt(ciphertext)

    return b64encode(ciphertext), b64encode(key), b64encode(nonce), decrypted_text

print(encrypt_Salsa20(b"This is Sixteen. AndAllIsWell."))