from Crypto.Hash import BLAKE2b

def blake2b_hash(data: str, digest_bits: int = 512) -> str:
    """
    Computes the BLAKE2b hash of the input data.

    Args:
        data (str): The input data to hash.
        digest_bits (int): The size of the output digest in bits (default: 512).

    Returns:
        str: The hexadecimal representation of the hash digest.
    """
    # Convert the input data to bytes
    byte_data = data.encode('utf-8')
    
    # Initialize the BLAKE2b hash object
    h_obj = BLAKE2b.new(digest_bits=digest_bits)
    
    # Update the hash object with the data
    h_obj.update(byte_data)
    
    # Retrieve the hexadecimal digest
    return h_obj.hexdigest()

# Example usage
if __name__ == "__main__":
    message = "Hello, world!"
    digest = blake2b_hash(message)
    print(f"Message: {message}")
    print(f"BLAKE2b Hash (Hex Digest): {digest}")
