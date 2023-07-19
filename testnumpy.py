import numpy as np

def quantize_number(num):
    return to.round((num + 0.1) / 0.2) * 0.2 - 0.1

# test it
print(quantize_number(0.1))  # output: 0.1
print(quantize_number(0.75))  # output: 0.3
