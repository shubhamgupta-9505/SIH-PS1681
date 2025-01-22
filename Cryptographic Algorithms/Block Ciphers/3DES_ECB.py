from Crypto.Cipher import DES3
from Crypto import Random
from Crypto.Util.Padding import pad, unpad
from base64 import b64encode

def generate_key_iv():
    """Generate a random 24-byte (192-bit) key for 3DES and an 8-byte IV."""
    # Ensure the key has correct parity bits
    while True:
        key = Random.get_random_bytes(24)  # 24 bytes for 3DES
        try:
            DES3.adjust_key_parity(key)
            break
        except ValueError:
            continue
    iv = Random.get_random_bytes(8)  # 8 bytes for 3DES IV
    return key, iv

def pcbc_encrypt(data, key, iv):
    """Encrypt data using 3DES with PCBC (Propagating Cipher Block Chaining)."""
    # Pad data to be a multiple of block size (8 bytes for DES)
    padded_data = pad(data.encode(), DES3.block_size)
    cipher = DES3.new(key, DES3.MODE_ECB)  # Use ECB as base mode for PCBC
    
    # Initialize the previous ciphertext block (initially IV)
    prev_block = iv
    encrypted_data = b""
    
    # Process each block
    for i in range(0, len(padded_data), DES3.block_size):
        block = padded_data[i:i + DES3.block_size]
        # XOR the plaintext block with the previous ciphertext
        xor_block = bytes(a ^ b for a, b in zip(block, prev_block))
        # Encrypt the XORed block
        encrypted_block = cipher.encrypt(xor_block)
        encrypted_data += encrypted_block
        # For PCBC, XOR both plaintext and ciphertext with next plaintext block
        prev_block = bytes(a ^ b for a, b in zip(block, encrypted_block))
    
    return encrypted_data

def encrypt_3des(data, mode,me):
    """
    Encrypt data using Triple DES in various modes.
    
    Args:
        data (str): The plaintext to encrypt
        mode (int): Encryption mode (1-6)
        
    Returns:
        str: Base64 encoded ciphertext
    """
    key, iv = generate_key_iv()
    
    try:
        if mode == 1:  # ECB
            cipher = DES3.new(key, DES3.MODE_ECB)
            encrypted_data = cipher.encrypt(pad(data.encode(), DES3.block_size))
        elif mode == 2:  # CBC
            cipher = DES3.new(key, DES3.MODE_CBC, iv)
            encrypted_data = cipher.encrypt(pad(data.encode(), DES3.block_size))
        elif mode == 3:  # CFB
            cipher = DES3.new(key, DES3.MODE_CFB, iv)
            encrypted_data = cipher.encrypt(data.encode())
        elif mode == 4:  # OFB
            cipher = DES3.new(key, DES3.MODE_OFB, iv)
            encrypted_data = cipher.encrypt(data.encode())
        elif mode == 5:  # CTR
            nonce = Random.get_random_bytes(4)
            cipher = DES3.new(key, DES3.MODE_CTR, nonce=nonce)
            encrypted_data = cipher.encrypt(data.encode())
        elif mode == 6:  # PCBC
            encrypted_data = pcbc_encrypt(data, key, iv)
        else:
            raise ValueError("Unsupported mode")
            
        encoded_data = b64encode(encrypted_data).decode('utf-8')
        
        # Print encryption parameters
        print(f"Mode: {me}")
        print("Key (hex):", key.hex())
        if mode in [2, 3, 4, 6]:  # Modes that use IV
            print("IV (hex):", iv.hex())
        elif mode == 5:  # CTR mode uses nonce
            print("Nonce (hex):", nonce.hex())
        print("Encrypted:", encoded_data)
        print('-' * 60)
        
        return encoded_data
        
    except Exception as e:
        print(f"Encryption error: {str(e)}")
        return None

data = input("Enter the data to encrypt: ")

# Encrypt data using different 3DES modes with numeric representation
encrypt_3des(data, 1,"ECB")  # Mode 1 -> ECB
encrypt_3des(data, 2,"CBC")  # Mode 2 -> CBC
encrypt_3des(data, 3,"CFB")  # Mode 3 -> CFB
encrypt_3des(data, 4,"OFB")  # Mode 4 -> OFB
encrypt_3des(data, 5,"CTR")  # Mode 5 -> CTR.
encrypt_3des(data, 6,"PCBC")  # Mode 5 -> PCBC.