'''
Created by Nikhat Singla on 11 Nov 2024
'''

def encrypt_Salsa20(plaintext: bytes, output_filename="salsa20_output.json"):
   
    from Crypto.Cipher import Salsa20
    from Crypto.Random import get_random_bytes
    from base64 import b64encode
    import json

    # Generate key and cipher
    key = get_random_bytes(32)
    cipher_encrypt = Salsa20.new(key=key)

    # Encrypt plaintext
    ciphertext = cipher_encrypt.encrypt(plaintext)
    nonce = cipher_encrypt.nonce

    # Decrypt to verify
    cipher_decrypt = Salsa20.new(key=key, nonce=nonce)
    decrypted_text = cipher_decrypt.decrypt(ciphertext)

    # Prepare the JSON output
    result_data = {
        "algorithm": "Salsa20",
        "key": b64encode(key).decode(),
        "nonce": b64encode(nonce).decode(),
        "ciphertext": b64encode(ciphertext).decode(),
        "decrypted_text": decrypted_text.decode()
    }

    # Save to JSON file
    with open(output_filename, "w") as json_file:
        json.dump(result_data, json_file, indent=4)

    return output_filename


# Example usage:
output_file = encrypt_Salsa20(b"This is Sixteen. AndAllIsWell.", "salsa20_output.json")
print(f"Results saved in {output_file}")
