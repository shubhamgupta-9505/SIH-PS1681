'''
Created on 09 Nov 2024 by Nikhat Singla

Following mode format is followed throughout:
Mode 1: ECB
Mode 2: CBC
Mode 3: CFB (doesn't need padded plaintext)
Mode 4: OFB (doesn't need padded plaintext)
Mode 5: CTR (doesn't need padded plaintext)
'''

from Crypto.Cipher import CAST
from Crypto.Random import get_random_bytes
from Crypto.Util.Padding import pad, unpad

def encrypt_CAST (plaintext: bytes, mode: int) -> dict:
    '''
    Description:
    This CAST implementation has 64 bit block size and uses 16 byte (recommended) key.

    Parameters:

    plaintext: a bytes representation of string (padded automatically)
    mode: an integer ranging from 1 to 5

    Returns:
    Dictionary containing key-value pairs for all necessities
    '''

    key_size = 16
    block_size = CAST.block_size

    ret_dict = {}
    ret_dict["algo"] = "CAST-128"
    ret_dict["algo_type"] = "BlockCipher"
    ret_dict["mode"] = mode

    plaintext = pad(plaintext, block_size)

    key = get_random_bytes(key_size)
    ret_dict["key"] = key.hex()

    if (mode == 1):
        cipher_encrypt = CAST.new(key, CAST.MODE_ECB)
        ciphertext = cipher_encrypt.encrypt(plaintext)

        cipher_decrypt = CAST.new(key, CAST.MODE_ECB)
        decrypted_text = cipher_decrypt.decrypt(ciphertext)
    elif (mode == 2):
        iv = get_random_bytes(block_size)
        ret_dict["iv"] = iv.hex()

        cipher_encrypt = CAST.new(key, CAST.MODE_CBC, iv)
        ciphertext = cipher_encrypt.encrypt(plaintext)

        cipher_decrypt = CAST.new(key, CAST.MODE_CBC, iv)
        decrypted_text = cipher_decrypt.decrypt(ciphertext)
    elif (mode == 3):
        iv = get_random_bytes(block_size)
        ret_dict["iv"] = iv.hex()

        segment_size = 8 # specifies segment size (in multiples of 8 bits)
        ret_dict["segment_size"] = segment_size

        cipher_encrypt = CAST.new(key, CAST.MODE_CFB, iv, segment_size=segment_size)
        ciphertext = cipher_encrypt.encrypt(plaintext)

        cipher_decrypt = CAST.new(key, CAST.MODE_CFB, iv, segment_size=segment_size)
        decrypted_text = cipher_decrypt.decrypt(ciphertext)
    elif (mode == 4):
        iv = get_random_bytes(block_size)
        ret_dict["iv"] = iv.hex()

        cipher_encrypt = CAST.new(key, CAST.MODE_OFB, iv)
        ciphertext = cipher_encrypt.encrypt(plaintext)

        cipher_decrypt = CAST.new(key, CAST.MODE_OFB, iv)
        decrypted_text = cipher_decrypt.decrypt(ciphertext)
    elif (mode == 5):
        nonce = get_random_bytes(int(block_size / 2))
        ret_dict["nonce"] = nonce.hex()

        cipher_encrypt = CAST.new(key, CAST.MODE_CTR, nonce=nonce)
        ciphertext = cipher_encrypt.encrypt(plaintext)

        cipher_decrypt = CAST.new(key, CAST.MODE_CTR, nonce=nonce)
        decrypted_text = cipher_decrypt.decrypt(ciphertext)
    
    ret_dict["ciphertext"] = ciphertext.hex()
    
    decrypted_text = unpad(decrypted_text, block_size)
    ret_dict["decrypted_text"] = decrypted_text.decode()
        
    return ret_dict

if __name__ == "__main__":
    print(encrypt_CAST(b"This is Sixteen. Not Sixteen.", 1))
    print(encrypt_CAST(b"This is Sixteen. Not Sixteen.", 2))
    print(encrypt_CAST(b"This is Sixteen. Not Sixteen.", 3))
    print(encrypt_CAST(b"This is Sixteen. Not Sixteen.", 4))
    print(encrypt_CAST(b"This is Sixteen. Not Sixteen.", 5))

