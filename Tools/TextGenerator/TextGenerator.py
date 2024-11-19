with open('full.txt', 'rt') as file:
    with open('short.txt', 'wt') as writefile:
        for _ in range(10):
            temp = file.read(1024 * 1024)
            writefile.write(temp)