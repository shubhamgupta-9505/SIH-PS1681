'''
Created by Nikhat Singla on 11 Nov 2024
'''

def encrypt_ARC4 (plaintext: bytes):
    '''
    Parameter "plaintext" is of type bytes.  Using recommended key length of 16 bytes.
    '''
    from Crypto.Cipher import ARC4
    from Crypto.Random import get_random_bytes
    from base64 import b64encode

    key = get_random_bytes(16)

    cipher_encrypt = ARC4.new(key)

    ciphertext = cipher_encrypt.encrypt(plaintext)

    cipher_decrypt = ARC4.new(key)
    decrypted_text = cipher_decrypt.decrypt(ciphertext)

    return b64encode(ciphertext), decrypted_text, b64encode(key)

print(encrypt_ARC4(b"This is Sixteen. AndAllIsWell."))
