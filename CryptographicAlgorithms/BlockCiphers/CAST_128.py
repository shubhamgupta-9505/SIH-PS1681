import json
from Crypto.Cipher import CAST
from Crypto.Random import get_random_bytes
from Crypto.Util import Counter
from base64 import b64encode

def encrypt_CAST(plaintext: bytes, mode: int, output_filename: str):
    '''
    Encrypts the given plaintext using CAST cipher in the specified mode.
    
    Args:
        plaintext (bytes): The plaintext data to be encrypted.
        mode (int): The mode of operation (1: ECB, 2: CBC, 3: CFB, 4: OFB, 5: CTR).
        output_filename (str): The filename for saving the JSON output.
    
    Returns:
        None: The result is written to a JSON file.
    '''
    key_size = 16
    block_size = 8

    # for CFB
    segment_size = 8

    # for CTR
    nonce_size = int(block_size / 2)
    counter_initial_value = 0

    key = get_random_bytes(key_size)
    iv = get_random_bytes(block_size)
    nonce = None

    if mode == 1:
        iv = None
        cipher_encrypt = CAST.new(key, CAST.MODE_ECB)
        ciphertext = cipher_encrypt.encrypt(plaintext)
        cipher_decrypt = CAST.new(key, CAST.MODE_ECB)
        decrypted_text = cipher_decrypt.decrypt(ciphertext)
    
    elif mode == 2:
        cipher_encrypt = CAST.new(key, CAST.MODE_CBC, iv)
        ciphertext = cipher_encrypt.encrypt(plaintext)
        cipher_decrypt = CAST.new(key, CAST.MODE_CBC, iv)
        decrypted_text = cipher_decrypt.decrypt(ciphertext)
    
    elif mode == 3:
        cipher_encrypt = CAST.new(key, CAST.MODE_CFB, iv, segment_size=segment_size)
        ciphertext = cipher_encrypt.encrypt(plaintext)
        cipher_decrypt = CAST.new(key, CAST.MODE_CFB, iv, segment_size=segment_size)
        decrypted_text = cipher_decrypt.decrypt(ciphertext)
    
    elif mode == 4:
        cipher_encrypt = CAST.new(key, CAST.MODE_OFB, iv)
        ciphertext = cipher_encrypt.encrypt(plaintext)
        cipher_decrypt = CAST.new(key, CAST.MODE_OFB, iv)
        decrypted_text = cipher_decrypt.decrypt(ciphertext)
    
    elif mode == 5:
        iv = None
        nonce = get_random_bytes(nonce_size)
        counter_1 = Counter.new(8 * (block_size - nonce_size), prefix=nonce, initial_value=counter_initial_value)
        cipher_encrypt = CAST.new(key, CAST.MODE_CTR, counter=counter_1)
        ciphertext = cipher_encrypt.encrypt(plaintext)
        counter_2 = Counter.new(8 * (block_size - nonce_size), prefix=nonce, initial_value=counter_initial_value)
        cipher_decrypt = CAST.new(key, CAST.MODE_CTR, counter=counter_2)
        decrypted_text = cipher_decrypt.decrypt(ciphertext)

    # Prepare the result data
    result_data = {
        "algorithm": "CAST",
        "key": b64encode(key).decode(),
        "iv": b64encode(iv).decode() if iv else None,
        "nonce": b64encode(nonce).decode() if nonce else None,
        "ciphertext": b64encode(ciphertext).decode(),
        "decrypted_text": decrypted_text.decode()
    }

    # Write the result data to a JSON file
    with open(output_filenameCAST, "w") as json_file:
        json.dump(result_data, json_file, indent=4)

    return output_filenameCAST

# Example usage:
encrypt_CAST(b"This is Sixteen.", 1, "cast_output_mode_1.json")
encrypt_CAST(b"This is Sixteen.", 2, "cast_output_mode_2.json")
encrypt_CAST(b"This is Sixteen.", 3, "cast_output_mode_3.json")
encrypt_CAST(b"This is Sixteen.", 4, "cast_output_mode_4.json")
encrypt_CAST(b"This is Sixteen.", 5, "cast_output_mode_5.json")
