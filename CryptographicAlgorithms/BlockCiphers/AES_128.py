import random
from Crypto.Cipher import AES
from Crypto.Random import get_random_bytes
from Crypto.Util import Counter
from base64 import b64encode
import json

def encrypt_AES(plaintext: bytes, mode: int, output_filename="aes_output.json"):
    '''
    Encrypts the plaintext with AES cipher in different modes.
    
    Args:
        plaintext (bytes): The data to be encrypted (should be a multiple of 16 bytes).
        mode (int): The mode of operation (1 = ECB, 2 = CBC, 3 = CFB, 4 = OFB, 5 = CTR).
        output_filename (str): The name of the output JSON file.
    
    Returns:
        dict: A dictionary with encryption details.
    '''
    key_size = 16
    block_size = 16

    # for CFB, OFB, CTR
    segment_size = 8  # specifies segment size (in multiples of 8 bits)
    nonce_size = int(block_size / 2)  # size of nonce (in bytes)
    counter_initial_value = 0  # initial value of counter

    key = get_random_bytes(key_size)
    iv = get_random_bytes(block_size)
    nonce = None
    ciphertext = None
    decrypted_text = None

    if mode == 1:  # ECB
        iv = None
        cipher_encrypt = AES.new(key, AES.MODE_ECB)
        ciphertext = cipher_encrypt.encrypt(plaintext)
        cipher_decrypt = AES.new(key, AES.MODE_ECB)
        decrypted_text = cipher_decrypt.decrypt(ciphertext)
        
    elif mode == 2:  # CBC
        cipher_encrypt = AES.new(key, AES.MODE_CBC, iv)
        ciphertext = cipher_encrypt.encrypt(plaintext)
        cipher_decrypt = AES.new(key, AES.MODE_CBC, iv)
        decrypted_text = cipher_decrypt.decrypt(ciphertext)
        
    elif mode == 3:  # CFB
        cipher_encrypt = AES.new(key, AES.MODE_CFB, iv, segment_size=segment_size)
        ciphertext = cipher_encrypt.encrypt(plaintext)
        cipher_decrypt = AES.new(key, AES.MODE_CFB, iv, segment_size=segment_size)
        decrypted_text = cipher_decrypt.decrypt(ciphertext)
        
    elif mode == 4:  # OFB
        cipher_encrypt = AES.new(key, AES.MODE_OFB, iv)
        ciphertext = cipher_encrypt.encrypt(plaintext)
        cipher_decrypt = AES.new(key, AES.MODE_OFB, iv)
        decrypted_text = cipher_decrypt.decrypt(ciphertext)
        
    elif mode == 5:  # CTR
        nonce = get_random_bytes(nonce_size)
        counter = Counter.new(8 * (block_size - nonce_size), prefix=nonce, initial_value=counter_initial_value)
        cipher_encrypt = AES.new(key, AES.MODE_CTR, counter=counter)
        ciphertext = cipher_encrypt.encrypt(plaintext)
        counter_2 = Counter.new(8 * (block_size - nonce_size), prefix=nonce, initial_value=counter_initial_value)
        cipher_decrypt = AES.new(key, AES.MODE_CTR, counter=counter_2)
        decrypted_text = cipher_decrypt.decrypt(ciphertext)
    
    # Prepare the result data in the desired JSON format
    result_data = {
        "algorithm": "AES",
        "key": b64encode(key).decode(),
        "iv": b64encode(iv).decode() if iv else None,
        "nonce": b64encode(nonce).decode() if nonce else None,
        "ciphertext": b64encode(ciphertext).decode(),
        "decrypted_text": decrypted_text.decode()
    }
    
    # Write the result data to a JSON file
    with open(output_filenameAes, "w") as json_file:
        json.dump(result_data, json_file, indent=4)
    
    return output_filenameAes

# Example usage:
print(encrypt_AES(b"This is Sixteen.", 1))  # ECB Mode
print(encrypt_AES(b"This is Sixteen.", 2))  # CBC Mode
print(encrypt_AES(b"This is Sixteen.", 3))  # CFB Mode
print(encrypt_AES(b"This is Sixteen.", 4))  # OFB Mode
print(encrypt_AES(b"This is Sixteen.", 5))  # CTR Mode
