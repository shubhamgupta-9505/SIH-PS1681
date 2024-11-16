import json
from hashlib import new

def jh_hash_to_json_file(message, digest_bits=256, output_filename="hash_output.json"):
    """
    Compute the JH hash of a given message and save it as a JSON file.

    Parameters:
        message (str): The input message to hash.
        digest_bits (int): The size of the digest in bits. Common values: 224, 256, 384, 512.
        output_filename (str): The name of the output JSON file.

    Returns:
        str: The path to the JSON file.
    """
    # Supported digest sizes for JH
    supported_bits = [224, 256, 384, 512]

    # Fallback to 256 bits if the input digest size is invalid
    if digest_bits not in supported_bits:
        print(f"Unsupported digest size '{digest_bits}'. Defaulting to 256 bits.")
        digest_bits = 256

    # Create a JH hash object using hashlib
    hash_name = f"JH-{digest_bits}"
    jh_hasher = new(hash_name)

    # Update the hasher with the message
    jh_hasher.update(message.encode('utf-8'))

    # Compute the hex digest
    jh_digest = jh_hasher.hexdigest()

    # Create the dictionary for the JSON output
    hash_data = {
        "algorithm": hash_name,
        "hexdigest": jh_digest
    }

    # Write the dictionary to a JSON file
    with open(output_filename, "w") as json_file:
        json.dump(hash_data, json_file, indent=4)

    # Return the file path
    return output_filename

