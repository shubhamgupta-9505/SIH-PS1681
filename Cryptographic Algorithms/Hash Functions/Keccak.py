''' 
Created on 15th November by Japnoor Kaur and Mahee Agarwal

This module implements the Keccak hash functions for the 64 bit word length (b=1600) and the fixed digest sizes of 224, 256, 384 and 512 bits.
''' 

from Crypto.Hash import SHA3_256

def create_keccak_object(data=b"", digest_bits=256, update_after_digest=False):
    """
    Create a Keccak hash object with optimized validation.

    Parameters:
        data (bytes): The initial chunk of the message to hash.
        digest_bits (int): The size of the digest in bits (224, 256, 384, 512).
        update_after_digest (bool): Whether digest() can be followed by update().

    Returns:
        Keccak_Hash: A Keccak hash object.
    """
    # Define valid digest sizes and their corresponding byte sizes
    valid_digests = {224: 28, 256: 32, 384: 48, 512: 64}
    
    # Ensure digest_bits is valid; fallback to default (256) if invalid
    digest_size = valid_digests.get(digest_bits, valid_digests[256])

    # Create Keccak object with the selected digest size
    keccak_obj = SHA3_256.new(digest_bytes=digest_size)

    # Add initial data if provided
    if data:
        keccak_obj.update(data)

    return keccak_obj

# Update keccak 
keccak_hash = create_keccak_object(data=initial_data, digest_bits=digest_bits, update_after_digest=True)

while True:
    # Accept input from the user
    data = input(">> ")

    # Exit the loop if the input is blank
    if not data:
        break

    # Update the hash object with the input
    keccak_obj.update(data.encode())  # Encode the input to bytes

# Get the final hash digest
final_digest = keccak_obj.hexdigest()
print("\nFinal Hash (hex):", final_digest)


