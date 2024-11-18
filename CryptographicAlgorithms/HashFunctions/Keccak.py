''' 
Created on 15th November by Japnoor Kaur and Mahee Agarwal.

This module implements the Keccak hash function.
''' 

from Crypto.Hash import SHA3_256

def encrypt_Keccak(message: bytes, digest_bits=256, update_after_digest=False):
    """
    Updates the Keccak hash object, and computes the final digest when input stops.

    Parameters:
        message (bytes): The message to hash
        digest_bits (int): The size of the digest in bits (224, 256, 384, 512).
        update_after_digest (bool): Whether digest() can be followed by update().

    Returns:
        str: Hexadecimal digest of the accumulated input.
    """
    # Define valid digest sizes and their corresponding byte sizes
    valid_digests = {224: 28, 256: 32, 384: 48, 512: 64}
    
    # Ensure digest_bits is valid; fallback to default (256) if invalid
    digest_size = valid_digests.get(digest_bits, valid_digests[256])

    # Create the Keccak hash object with the given parameters
    keccak_obj = SHA3_256.new(digest_bytes=digest_size)
    
    keccak_obj.update(message)

    # Return the hexadecimal digest of the accumulated input
    return keccak_obj.hexdigest()

