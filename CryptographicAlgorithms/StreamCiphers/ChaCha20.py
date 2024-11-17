'''
Created by Nikhat Singla on 10 Nov 2024
'''


def encrypt_ChaCha20(plaintext: bytes, output_filename="chacha20_output.json"):
    
    from Crypto.Cipher import ChaCha20
    from Crypto.Random import get_random_bytes
    from base64 import b64encode
    import json

    # Generate a random 32-byte key
    key = get_random_bytes(32)

    # Initialize ChaCha20 cipher for encryption
    cipher_encrypt = ChaCha20.new(key=key)
    nonce = cipher_encrypt.nonce

    # Encrypt plaintext
    ciphertext = cipher_encrypt.encrypt(plaintext)

    # Initialize ChaCha20 cipher for decryption to verify
    cipher_decrypt = ChaCha20.new(key=key, nonce=nonce)
    decrypted_text = cipher_decrypt.decrypt(ciphertext)

    # Prepare the JSON output
    result_data = {
        "algorithm": "ChaCha20",
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
output_file = encrypt_ChaCha20(b"This is Sixteen. AndAllIsWell.", "chacha20_output.json")
print(f"Results saved in {output_file}")
