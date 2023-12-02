import copy

all_seq = []
lb = [0, 1]
ub = [3, 5]


def generate(seq):
    i = len(seq) - 1
    while i >= 0:
        i = len(seq) - 1
        while i >= 0 and seq[i] == ub[i]:
            i -= 1
        if i >= 0:
            seq[i] += 1
            seq[i+1:] = lb[i+1:]


generate(copy.deepcopy(lb))