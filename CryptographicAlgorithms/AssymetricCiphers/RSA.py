''' 
Created on 15 November 2024 by Japnoor Kaur

RSA is used for both confidentiality (encryption) and authentication (digitalsignature).
Signing and decryption are significantly slower than verification and encryption.

Sufficient length of the RSA modulus is 3072 bits
'''

import random
from math import gcd
from sympy import mod_inverse  # Ensure you have sympy installed: pip install sympy
p=random.getrandbits(16)
q=random.getrandbits(16)

'Generate new RSA keys'
def generate_keypair(p, q):
    
    while p==q :
       q=random.getrandbits(16)
    
    n = p * q
    f = (p - 1) * (q - 1)
    
    e = 65537  # Fixed common public exponent
    while(gcd(e,f)!=1):
        p=random.getrandbits(16)
        q=random.getrandbits(16)
        while p==q :
            q.getrandbits(16)
        n=p*q
        f=(p-1)*(q-1)
        
    #Generating private key
    d = mod_inverse(e, f)
    
    return ((e, n), (d, n))
    
from Crypto.PublicKey import RSA
from Crypto.Random import get_random_bytes

'Exporting RSA keys'
def export_key(format='PEM', passphrase=None, pkcs=1, protection=None, prot_params=None):
    # Generate a new RSA key pair in a tuple t where t[0]=public key and t[1]=private key
    t=generate_keypair(p,q)

    # Export the private key
    private_key = t[1].export_key(
        format=format,
        passphrase=passphrase,
        pkcs=pkcs,
        protection=protection,
        prot_params=prot_params,
        randfunc=get_random_bytes
    )

    # Export the public key
    public_key = t[0].export_key(format=format)

    # Return the keys in the chosen format
    return private_key, public_key
    
'Encrypt the key'
from Crypto.Cipher import PKCS1_OAEP
from Crypto.Hash import SHA256

def encrypt_message(public_key, message):
    """
    Encrypts a message using RSA with PKCS#1 OAEP.

    Parameters:
        public_key (bytes): The RSA public key in PEM format.
        message (bytes): The plaintext message to encrypt.

    Returns:
        bytes: The encrypted ciphertext.

    Raises:
        ValueError: If the message is too long to be encrypted with the given RSA key.
    """
    # Import the public key
    rsa_key = RSA.import_key(public_key)

    # Create a cipher object using PKCS1_OAEP with SHA-256
    cipher = PKCS1_OAEP.new(key=rsa_key, hashAlgo=SHA256)

    # Calculate the maximum message length
    key_size_in_bytes = rsa_key.size_in_bytes()  # RSA modulus size in bytes
    hash_size_in_bytes = SHA256.digest_size  # SHA-256 hash output size in bytes
    max_message_length = key_size_in_bytes - 2 - 2 * hash_size_in_bytes

    # Encrypt the message in chunks
    chunks = [
        cipher.encrypt(message[i:i + max_message_length])
        for i in range(0, len(message), max_message_length)
    ]

    # Concatenate all encrypted chunks
    return b"".join(chunks)

'Decrypt the key'
from Crypto.Cipher import PKCS1_v1_5

def decrypt_message(private_key, ciphertext, sentinel=None):
    """
    Decrypts a PKCS#1 v1.5 ciphertext using the given private RSA key.

    Parameters:
        private_key (bytes): The RSA private key in PEM format.
        ciphertext (bytes): The encrypted message to decrypt.
        sentinel (any): The value to return in case decryption fails.

    Returns:
        bytes: The decrypted plaintext message, or the sentinel value if decryption fails.
    """
    # Import the private key
    rsa_key = RSA.import_key(private_key)

    # Create a cipher object for PKCS#1 v1.5
    cipher = PKCS1_v1_5.new(rsa_key)

    # Decrypt the ciphertext
    plaintext = cipher.decrypt(ciphertext, sentinel)
    return plaintext


