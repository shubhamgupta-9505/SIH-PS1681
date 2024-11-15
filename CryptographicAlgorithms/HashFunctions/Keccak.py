from Crypto.Hash import SHA3_256

def keccak_hash_from_inputs(digest_bits=256, update_after_digest=False):
    """
    Continuously takes input from the user, updates the Keccak hash object,
    and computes the final digest when input stops.

    Parameters:
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

    # Return the hexadecimal digest of the accumulated input
    return keccak_obj.hexdigest()

