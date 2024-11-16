from hashlib import new

def jh_hash(message, digest_bits=256):
    """
    Compute the JH hash of a given message.

    Parameters:
        message (str): The input message to hash.
        digest_bits (int): The size of the digest in bits. Common values: 224, 256, 384, 512.

    Returns:
        str: Hexadecimal hash of the input message.
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

    jh_digest= jh_hasher.hexdigest()

    import json
    # Convert to key-value pairs (2 characters per value)
    key_value_pairs = {f"key{i}": jh_digest[i:i+2] for i in range(0, len(hexdigest), 2)}

    # Save to JSON file
    json_filename = "digest_key_value.json"
    with open(json_filename, "w") as json_file:
        json.dump(key_value_pairs, json_file, indent=4)  # Indent for readability

    # Return the json file 
    return json_filename 


