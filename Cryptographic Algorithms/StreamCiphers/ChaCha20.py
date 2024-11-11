'''
Created by Nikhat Singla on 10 Nov 2024
'''

def encrypt_ChaCha20 (plaintext: bytes):
    '''
    Parameter "plaintext" is of type bytes. 32 byte key and 8 bytes nonce is used (default value).
    '''
    from Crypto.Cipher import ChaCha20
    from Crypto.Random import get_random_bytes
    from base64 import b64encode

    key = get_random_bytes(32)

    cipher_encrypt = ChaCha20.new(key=key)
    nonce = cipher_encrypt.nonce

    ciphertext = cipher_encrypt.encrypt(plaintext)

    cipher_decrypt = ChaCha20.new(key=key, nonce=nonce)
    decrypted_text = cipher_decrypt.decrypt(ciphertext)

    return b64encode(ciphertext), decrypted_text, b64encode(key), b64encode(nonce)

print(encrypt_ChaCha20(b"This is Sixteen. AndAllIsWell."))