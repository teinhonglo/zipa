
import sys

if len(sys.argv) != 3:
    print(f"Error, {sys.argv[0]} <train_text> <train_units>")
    sys.exit(1)

train_text = sys.argv[1]
units_file = sys.argv[2]

units = {}
with open(train_text, 'r') as fin:   
    line = fin.readline()
    while line:
        line = line.strip().split(' ')
        for char in line[1:]:
            try:
                if units[char] == True:
                    continue
            except:
                units[char] = True
        line = fin.readline()

# NOTE: This is important to keep phoneme vocab the same !!!
units = sorted(units)

fwriter = open(units_file, 'w')
for char in units:
    print(char, file=fwriter)
