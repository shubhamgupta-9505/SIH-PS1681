''' 
Created on 15th November by Mahee Agarwal.

This module implements the Blake hash function and outputs results in a JSON file.
''' 

import json
from Crypto.Hash import BLAKE2b

def encrypt_BLAKE_to_json(data: str, digest_bits: int = 512, output_filename="blake_hash_output.json") -> str:
    """
    Computes the BLAKE2b hash of the input data, saves it in a JSON file, and returns the file path.

    Args:
        data (str): The input data to hash.
        digest_bits (int): The size of the output digest in bits (default: 512).
        output_filename (str): The name of the JSON file to save the hash output.

    Returns:
        str: The path to the JSON file containing the hash output.
    """
    # Convert the input data to bytes
    byte_data = data.encode('utf-8')
    
    # Initialize the BLAKE2b hash object
    h_obj = BLAKE2b.new(digest_bits=digest_bits)
    
    # Update the hash object with the data
    h_obj.update(byte_data)
    
    # Retrieve the hexadecimal digest
    blake_digest = h_obj.hexdigest()

    # Create a dictionary for the JSON output
    hash_data = {
        "algorithm": f"BLAKE2b-{digest_bits}",
        "hexdigest": blake_digest
    }

    # Write the dictionary to a JSON file
    with open(output_filenameblake, "w") as json_file:
        json.dump(hash_data, json_file, indent=4)

    # Return the path to the JSON file
    return output_filenameblake
