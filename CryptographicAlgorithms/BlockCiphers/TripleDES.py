import json
from Crypto.Cipher import DES3
from Crypto.Random import get_random_bytes
from Crypto.Util import Counter
from base64 import b64encode

def encrypt_DES3(plaintext: bytes, mode: int, output_filename: str):
    '''
    Encrypts the given plaintext using DES3 cipher in the specified mode.
    
    Args:
        plaintext (bytes): The plaintext data to be encrypted.
        mode (int): The mode of operation (1: ECB, 2: CBC, 3: CFB, 4: OFB, 5: CTR).
        output_filename (str): The filename for saving the JSON output.
    
    Returns:
        None: The result is written to a JSON file.
    '''
    key_size = 8 * 3  # DES3 key size (3 DES keys)
    block_size = 8    # DES block size

    # for CFB
    segment_size = 8  # specifies segment size (in multiples of 8 bits)

    # for CTR
    nonce = None  # specifies nonce (set to None due to other modes of operations)
    nonce_size = int(block_size / 2)  # specifies size of nonce (in bytes)
    counter_initial_value = 0  # specifies initial value of counter

    key = get_random_bytes(key_size)
    iv = get_random_bytes(block_size)

    if mode == 1:  # ECB
        iv = None
        cipher_encrypt = DES3.new(key, DES3.MODE_ECB)
        ciphertext = cipher_encrypt.encrypt(plaintext)
        cipher_decrypt = DES3.new(key, DES3.MODE_ECB)
        decrypted_text = cipher_decrypt.decrypt(ciphertext)

    elif mode == 2:  # CBC
        cipher_encrypt = DES3.new(key, DES3.MODE_CBC, iv)
        ciphertext = cipher_encrypt.encrypt(plaintext)
        cipher_decrypt = DES3.new(key, DES3.MODE_CBC, iv)
        decrypted_text = cipher_decrypt.decrypt(ciphertext)

    elif mode == 3:  # CFB
        cipher_encrypt = DES3.new(key, DES3.MODE_CFB, iv, segment_size=segment_size)
        ciphertext = cipher_encrypt.encrypt(plaintext)
        cipher_decrypt = DES3.new(key, DES3.MODE_CFB, iv, segment_size=segment_size)
        decrypted_text = cipher_decrypt.decrypt(ciphertext)

    elif mode == 4:  # OFB
        cipher_encrypt = DES3.new(key, DES3.MODE_OFB, iv)
        ciphertext = cipher_encrypt.encrypt(plaintext)
        cipher_decrypt = DES3.new(key, DES3.MODE_OFB, iv)
        decrypted_text = cipher_decrypt.decrypt(ciphertext)

    elif mode == 5:  # CTR
        iv = None
        nonce = get_random_bytes(nonce_size)
        counter_1 = Counter.new(8 * (block_size - nonce_size), prefix=nonce, initial_value=counter_initial_value)
        cipher_encrypt = DES3.new(key, DES3.MODE_CTR, counter=counter_1)
        ciphertext = cipher_encrypt.encrypt(plaintext)
        counter_2 = Counter.new(8 * (block_size - nonce_size), prefix=nonce, initial_value=counter_initial_value)
        cipher_decrypt = DES3.new(key, DES3.MODE_CTR, counter=counter_2)
        decrypted_text = cipher_decrypt.decrypt(ciphertext)

    # Prepare the result data
    result_data = {
        "algorithm": "DES3",
        "key": b64encode(key).decode(),
        "iv": b64encode(iv).decode() if iv else None,
        "nonce": b64encode(nonce).decode() if nonce else None,
        "ciphertext": b64encode(ciphertext).decode(),
        "decrypted_text": decrypted_text.decode()
    }

    # Write the result data to a JSON file
    with open(output_filenametripleDES, "w") as json_file:
        json.dump(result_data, json_file, indent=4)

    return output_filenametripleDES

# Example usage:
encrypt_DES3(b"This is Sixteen.", 1, "des3_output_mode_1.json")
encrypt_DES3(b"This is Sixteen.", 2, "des3_output_mode_2.json")
encrypt_DES3(b"This is Sixteen.", 3, "des3_output_mode_3.json")
encrypt_DES3(b"This is Sixteen.", 4, "des3_output_mode_4.json")
encrypt_DES3(b"This is Sixteen.", 5, "des3_output_mode_5.json")
