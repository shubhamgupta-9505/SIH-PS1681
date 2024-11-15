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

    # Return the hexadecimal digest
    return jh_hasher.hexdigest()


