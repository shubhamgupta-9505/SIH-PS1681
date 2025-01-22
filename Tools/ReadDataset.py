'''
Created by Nikhat Singla on 20 Nov 2024
'''

def read_until_delimiter(file, delimiter):
    result = []
    while True:
        char = file.read(1)  # Read one character at a time
        if not char or char == delimiter:  # Stop if end of file or delimiter is found
            break
        result.append(char)
    return b''.join(result)

# Define the delimiter
delimiter = b','

with open('Datasets/Encrypted/encrypted_nsv.nsv', 'rb') as file:
    file_data = []

    while True:
        col = []

        # Read 4 columns
        for i in range(4):
            data = read_until_delimiter(file, delimiter)
            if not data:  # If no data is read, end of file is reached
                break
            col.append(data)

        if not col:  # If col is empty, break the loop
            break

        # Read the ciphertext length and then the ciphertext itself
        ciphertext_length = int(col[3])
        ciphertext_bytes = file.read(ciphertext_length)
        col.append(ciphertext_bytes.hex())

        # print(col)
        file_data.append(col)

        # Read the newline character (if any)
        file.read(1)  # for \n

print(file_data[25024])