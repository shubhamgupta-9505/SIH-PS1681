'''
Created on 15th November by Japnoor Kaur and Mahee Agarwal.

This module implements the Keccak hash function and outputs results in a JSON file.
'''

import json
from Crypto.Hash import SHA3_256

def keccak_hash_to_json_file(digest_bits=256, update_after_digest=False, output_filename="keccak_hash_output.json"):
    """
    Continuously takes input from the user, updates the Keccak hash object,
    computes the final digest when input stops, and saves the result in a JSON file.

    Parameters:
        digest_bits (int): The size of the digest in bits (224, 256, 384, 512).
        update_after_digest (bool): Whether digest() can be followed by update().
        output_filename (str): The name of the output JSON file.

    Returns:
        str: Path to the JSON file containing the hash output.
    """
    # Define valid digest sizes and their corresponding byte sizes
    valid_digests = {224: 28, 256: 32, 384: 48, 512: 64}
    
    # Ensure digest_bits is valid; fallback to default (256) if invalid
    digest_size = valid_digests.get(digest_bits, valid_digests[256])

    # Create the Keccak hash object with the given parameters
    keccak_obj = SHA3_256.new(digest_bytes=digest_size)
    
    print("Enter data to hash (press Enter on an empty line to stop):")

    while True:
        # Take input from the user
        user_input = input(">> ")
        
        # Stop if input is empty
        if not user_input:
            break
        
        # Update the hash object with the new input
        keccak_obj.update(user_input.encode('utf-8'))

    # If update_after_digest is True, allow further updates after digest
    if update_after_digest:
        print("Digest created. You can still update the hash object.")
        while True:
            user_input = input(">> ")
            if not user_input:
                break
            keccak_obj.update(user_input.encode('utf-8'))

    # Compute the hexadecimal digest of the accumulated input
    keccak_digest = keccak_obj.hexdigest()

    # Create a dictionary for the JSON output
    hash_data = {
        "algorithm": f"Keccak-{digest_bits}",
        "hexdigest": keccak_digest
    }

    # Write the dictionary to a JSON file
    with open(output_filenamekeccak, "w") as json_file:
        json.dump(hash_data, json_file, indent=4)

    # Return the path to the JSON file
    return output_filenamekeccak
