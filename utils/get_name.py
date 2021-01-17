from string import digits


def correction_name(name):
    remove_digits = str.maketrans('', '', digits)
    res = name.translate(remove_digits)
    return res
