'''
Created on 19 Nov 2024 by Nikhat Singla

This module implements the JH hash function.
'''

import hashlib

def hash_JH(message: bytes, digest_bits=256) -> dict:
    """
    Computes the JH hash of a given message.

    Parameters:
        message (bytes): The input message to hash (in bytes format).
        digest_bits (int): The size of the digest in bits. Common values: 224, 256 (default), 384, 512.

    Returns:
        Dictionary containing key-value pairs associated with digest
    """
    
    # Supported digest sizes for JH
    supported_bits = [224, 256, 384, 512]

    # Fallback to 256 bits if the input digest size is invalid
    if digest_bits not in supported_bits:
        print(f"Unsupported digest size '{digest_bits}'. Defaulting to 256 bits.")
        digest_bits = 256

    # Create a JH hash object using hashlib
    hash_name = f"JH-{digest_bits}"
    jh_hasher = hashlib.new(hash_name)

    # Update the hasher with the message
    jh_hasher.update(message)

    # Return the hexadecimal digest
    return jh_hasher.digest()

# Example usage
if __name__ == "__main__":
    message = b"Hello, world!"
    # digest = hash_JH(message)
    print(f"Message: {message}")
    # print(f"JH Hash: {digest}")

    print(hashlib.algorithms_available)
