from Crypto.Random import get_random_bytes

from Crypto.Cipher import DES3
from Crypto.Cipher import CAST

key = get_random_bytes(24)
print(key.hex(' '))
while True:
    try:
        key = DES3.adjust_key_parity(key)
        break
    except ValueError:
        pass
print(key.hex(' '))
print(CAST.block_size)

#LEFT
'''
JH_2
JH
Keccak


ARC4
ChaCha20
Salsa20


RSA
'''