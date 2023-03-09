
def append_to_file(path: str, lines: list):
    with open(path,'a') as f:
        for line in lines:
            f.write(line)
            f.write('\n')
