'''
Created on 09 Nov 2024 by Nikhat Singla

Following mode format is followed throughout:
Mode 1: ECB
Mode 2: CBC
Mode 3: CFB (doesn't need padded plaintext)
Mode 4: OFB (doesn't need padded plaintext)
Mode 5: CTR (doesn't need padded plaintext)
'''

def encrypt_AES (plaintext: bytes, mode: int):
    '''
    Input parameter "plaintext" must be in Bytes format, with bytes divisible by AES block size (128 bits)
    '''

    key_size = 16
    block_size = 16

    # for CFB
    segment_size = 8 # specifies segment size (in multiples of 8 bits)

    # for CTR
    nonce = None # specifies nonce (set to None due to other modes of operations)
    nonce_size = int(block_size / 2) # specifies size of nonce (in bytes)
    counter_initial_value = 0 # specifies initial value of counter

    from Crypto.Cipher import AES
    from Crypto.Random import get_random_bytes
    from base64 import b64encode, b64decode

    key = get_random_bytes(key_size)
    iv = get_random_bytes(block_size)

    if (mode == 1):
        iv = None
        cipher_encrypt = AES.new(key, AES.MODE_ECB)
        ciphertext = cipher_encrypt.encrypt(plaintext)

        cipher_decrypt = AES.new(key, AES.MODE_ECB)
        decrypted_text = cipher_decrypt.decrypt(ciphertext)
    elif (mode == 2):
        cipher_encrypt = AES.new(key, AES.MODE_CBC, iv)
        ciphertext = cipher_encrypt.encrypt(plaintext)

        cipher_decrypt = AES.new(key, AES.MODE_CBC, iv)
        decrypted_text = cipher_decrypt.decrypt(ciphertext)
    elif (mode == 3):
        cipher_encrypt = AES.new(key, AES.MODE_CFB, iv, segment_size=segment_size)
        ciphertext = cipher_encrypt.encrypt(plaintext)

        cipher_decrypt = AES.new(key, AES.MODE_CFB, iv, segment_size=segment_size)
        decrypted_text = cipher_decrypt.decrypt(ciphertext)
    elif (mode == 4):
        cipher_encrypt = AES.new(key, AES.MODE_OFB, iv)
        ciphertext = cipher_encrypt.encrypt(plaintext)

        cipher_decrypt = AES.new(key, AES.MODE_OFB, iv)
        decrypted_text = cipher_decrypt.decrypt(ciphertext)
    elif (mode == 5):
        from Crypto.Util import Counter

        iv = None
        nonce = get_random_bytes(nonce_size)
        counter_1 = Counter.new(8 * (block_size - nonce_size), prefix=nonce, initial_value=counter_initial_value)
        cipher_encrypt = AES.new(key, AES.MODE_CTR, counter=counter_1)
        ciphertext = cipher_encrypt.encrypt(plaintext)

        counter_2 = Counter.new(8 * (block_size - nonce_size), prefix=nonce, initial_value=counter_initial_value)
        cipher_decrypt = AES.new(key, AES.MODE_CTR, counter=counter_2)
        decrypted_text = cipher_decrypt.decrypt(ciphertext)
        
    return b64encode(key), iv if (iv == None) else b64encode(iv), nonce if (nonce == None) else b64encode(nonce), b64encode(ciphertext), decrypted_text

if __name__ == "__main__":
    print(encrypt_AES(b"This is Sixteen.", 1))
    print(encrypt_AES(b"This is Sixteen.", 2))
    print(encrypt_AES(b"This is Sixteen.", 3))
    print(encrypt_AES(b"This is Sixteen.", 4))
    print(encrypt_AES(b"This is Sixteen.", 5))

