import os

paths = 'data/unpair/DICM'
output_file = '/data/unpair/DICM/test.txt'

def sort_filenames(filename):

    digits = ''.join(filter(str.isdigit, filename))
    if digits:
        return int(digits)
    else:
        return float('inf')  

filenames = sorted(os.listdir(paths), key=sort_filenames)

with open(output_file, 'w') as f:
    for filename in filenames:
        if os.path.splitext(filename)[1] == '.jpg' or os.path.splitext(filename)[1] == '.JPG':
            f.write(filename + '\n')
f.close()
