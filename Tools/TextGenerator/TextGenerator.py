'''
Created by Nikhat Singla on 19 Nov 2024
'''

size = 1 # in MiB
with open('Tools/TextGenerator/full.txt', 'rt') as file:
    with open(f'Datasets/{size}MiB.txt', 'wt') as writefile:
        for _ in range(size):
            temp = file.read(1024 * 1024)
            writefile.write(temp)