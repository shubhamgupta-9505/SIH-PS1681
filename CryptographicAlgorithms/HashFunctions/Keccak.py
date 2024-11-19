''' 
Created on 15th November 2024 by Japnoor Kaur, Mahee Agarwal and Nikhat Singla.

This module implements the Keccak hash function.
''' 

from Crypto.Hash import keccak

def hash_KECCAK(message: bytes, digest_bits=256) -> dict:
    """
    Hashes the message using Keccak hash function.

    Parameters:
        message (bytes): The message to hash
        digest_bits (int): The size of the digest in bits (224, 256 (default), 384, 512).

    Returns:
        Dictionary containing digest of the input message.
    """
    
    # Define valid digest sizes and their corresponding byte sizes
    valid_digests = {224: 28, 256: 32, 384: 48, 512: 64}
    
    # Ensure digest_bits is valid; fallback to default (256) if invalid
    digest_size = valid_digests.get(digest_bits, valid_digests[256])

    # Create the Keccak hash object with the given parameters
    keccak_obj = keccak.new(digest_bytes=digest_size)
    
    keccak_obj.update(message)

    # Return the digest of the accumulated input
    return {"algo": "Keccak", "algo_type": "HashFun", "digest": keccak_obj.digest()}

# Example usage
if __name__ == "__main__":
    message = b"Hello, world!"
    digest = hash_KECCAK(message)
    print(f"Message: {message}")
    print(f"Keccak Hash: {digest}")

