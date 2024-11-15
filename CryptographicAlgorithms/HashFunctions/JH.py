def leftrotate(n, b):
    """Left rotate a 64-bit integer n by b bits."""
    return ((n << b) | (n >> (64 - b))) & 0xffffffffffffffff

def JH_hash(message, hash_size=512):
    """
    JH Hash Function (Simplified for Educational Purposes)
    
    :param message: Input message to be hashed (bytes)
    :param hash_size: Output hash size in bits (default: 512)
    :return: JH Hash of the input message (bytes)
    """
    # Constants and Initializations
    h = [0x6a09e667f3bcc908] * 8  # Initial hash values
    message = bytearray(message)
    message_len = len(message) * 8  # Convert length to bits
    message += b'\x01'  # Append the '1' bit
    while (len(message) * 8) % 1024 != 896:  # 1024-bit (128 byte) block size, pad to 896 bits
        message += b'\x00'
    message += message_len.to_bytes(8, 'big')  # Append the original length

    # Process Message in 1024-bit (128 byte) Blocks
    for i in range(0, len(message), 128):
        block = message[i:i+128]
        # Convert block to 64-bit integers (big-endian)
        w = [int.from_bytes(block[j:j+8], 'big') for j in range(0, 128, 8)]

        # JH Mix Function (Simplified, actual JH has more complex mixing)
        for t in range(42):  # Simplified round count for illustration
            v = h + w
            for j in range(8):
                v[j] = leftrotate(v[j] ^ v[(j+1)%8] ^ v[(j+4)%8], 3)  # Simplified rotation/mixing
            h, w = w, v  # Swap for next round

        # Update Hash State
        for j in range(8):
            h[j] ^= w[j]

    # Output Hash
    output_hash = b''
    for i in range(hash_size // 64):  # Convert hash state to bytes
        output_hash += h[i % 8].to_bytes(8, 'big')
    return output_hash[:hash_size // 8]  # Trim to desired hash size

# Example Usage
if __name__ == "__main__":
    message = b"Hello, World!"
    hash_output = JH_hash(message)
    print(f"JH Hash ({len(hash_output)*8} bits) of '{message.decode()}'")
    print(hash_output.hex())
