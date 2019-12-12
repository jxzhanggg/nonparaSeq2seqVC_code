
def read_text(fn):
    text = []
    with open(fn) as f:
        lines = f.readlines()
        for line in lines:
            start, end, phone = line.strip().split()
            text.append([int(start), int(end), phone])
    return text