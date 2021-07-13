
import subprocess
import sys

#Arguments
#Output file name,

with open(sys.argv[1], "w+") as output:
    subprocess.call(["python3", "BacteriumDiversity/simulation-bigger.py"] + sys.argv[2:], stdout=output);
#

#with open("cora.txt", "w+") as output:
#    subprocess.call(["python3", "BacteriumDiversity/cora.py"], stdout=output);