#Python script to replace all instances of function evaluation into constants
import math
import re
from math import pow

def float_compare(a, b, err):
    if math.fabs(a - b) > err:
        return False
    return True

def replacer(string):
    """Replaces all instances of number * pow(base, exp) with number in the string.

    >>> test1 = "4.555*pow(10, -2)"
    >>> test2 = "45454 454 453 434 3 44, 3.444*pow(10, -3)"
    >>> float_compare(float(replacer(test1)), 4.55 *10 **-2, .01)
    True
    >>> "pow" not in replacer(test2)
    True
    >>> test3 = test1 + " 3.44*pow(10, -32)"
    >>> test3dat = replacer(test3)
    >>> "pow" not in test3dat
    True
    >>> data = test3dat.split(" ")
    >>> float_compare(float(data[0]), 4.55 *10 **-2, .01)
    True
    >>> float_compare(float(data[1]), 3.44*10**-32, .01)
    True
    """
    pt = re.compile(r"(\d\.\d+)\*pow\(\d{2}, (-\d{0,2})\)")
    while re.search(pt, string):
        arr = list(map(float, re.search(pt, string).groups()))
        string = re.sub(pt, "{0:.25f}".format(arr[0] * 10**arr[1]), string, count=1) 
    return string 

def main():
    with open("data.h", mode = "r") as f:
        with open("new_data.h", mode = "w") as w:
            for line in f:
                w.write(replacer(line))
    return

if __name__ == "__main__":
    import doctest
    doctest.testmod()
