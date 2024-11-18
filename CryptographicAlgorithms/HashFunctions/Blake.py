''' 
Created on 15th November by Mahee Agarwal.

This module implements the Blake2b hash function.
''' 

from Crypto.Hash import BLAKE2b

def encrypt_BLAKE(data: bytes, digest_bits: int = 512) -> dict:
    """
    Computes the BLAKE2b hash of the input data.

    Args:
        data (bytes): The input data (in bytes format) to hash.
        digest_bits (int): The size of the output digest in bits (default: 512).

    Returns:
        dict: Contains the hash digest in bytes format.
    """
    
    # Initialize the BLAKE2b hash object
    h_obj = BLAKE2b.new(digest_bits=digest_bits)
    
    # Update the hash object with the data
    h_obj.update(data)
    
    # Retrieve the hexadecimal digest
    return {"digest": h_obj.digest()}

# Example usage
if __name__ == "__main__":
    message = b"Hello, world!"
    digest = encrypt_BLAKE(message)
    print(f"Message: {message}")
    print(f"BLAKE2b Hash: {digest}")
