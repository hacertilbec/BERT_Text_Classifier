#https://gist.github.com/canerbasaran/6334104
from random import randint


def rastgele_tc():
    tcno = str(randint(100000000, 1000000000))
    list_tc = list(map(int, tcno))
    tc10 = (sum(list_tc[::2]) * 7 - sum(list_tc[1::2])) % 10
    return tcno + str(tc10) + str((sum(list_tc[:9]) + tc10) % 10)