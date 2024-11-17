def encrypt_ARC4(plaintext: bytes, output_filename="arc4_output.json"):
   
    from Crypto.Cipher import ARC4
    from Crypto.Random import get_random_bytes
    from base64 import b64encode
    import json

    # Generate a random 16-byte key
    key = get_random_bytes(16)

    # Initialize ARC4 cipher for encryption
    cipher_encrypt = ARC4.new(key)
    ciphertext = cipher_encrypt.encrypt(plaintext)

    # Initialize ARC4 cipher for decryption to verify
    cipher_decrypt = ARC4.new(key)
    decrypted_text = cipher_decrypt.decrypt(ciphertext)

    # Prepare the JSON output
    result_data = {
        "algorithm": "ARC4",
        "key": b64encode(key).decode(),
        "ciphertext": b64encode(ciphertext).decode(),
        "decrypted_text": decrypted_text.decode()
    }

    # Save to JSON file
    with open(output_filename, "w") as json_file:
        json.dump(result_data, json_file, indent=4)

    return output_filename


# Example usage:
output_file = encrypt_ARC4(b"This is Sixteen. AndAllIsWell.", "arc4_output.json")
print(f"Results saved in {output_file}")

