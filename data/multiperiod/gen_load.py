#!/usr/bin/env python

import sys
import subprocess

def usage():
    print('Usage: python gen_load.py case scalefile')
    print('  case     : case name')
    print('  scalefile: filename containing scaling factors')

if len(sys.argv) < 3:
    usage()
    sys.exit(-1)

case = sys.argv[1]
scalefile = sys.argv[2]

casefile = 'data/' + case + '.bus'
ps1 = subprocess.Popen(['sed', '-n', '$=', casefile], stdout=subprocess.PIPE)
num_bus = int(subprocess.check_output(['awk', '{print $1}'], stdin=ps1.stdout).strip())

pd = [0]*num_bus
qd = [0]*num_bus

f = open(casefile, 'r')
for i, row in enumerate(f):
    cols = row.split()
    pd[i] = float(cols[2].strip())
    qd[i] = float(cols[3].strip())
f.close()

ps1 = subprocess.Popen(['sed', '-n', '$=', scalefile], stdout=subprocess.PIPE)
num_scales = int(subprocess.check_output(['awk', '{print $1}'], stdin=ps1.stdout).strip())

scales = [0]*num_scales

f = open(scalefile, 'r')
for i, row in enumerate(f):
    scales[i] = float(row.strip())
f.close()

prefix = 'data/' + case + '_onehour_' + str(num_scales)
pdout = open(prefix + '.Pd', 'w')
qdout = open(prefix + '.Qd', 'w')

for i in range(0, num_bus):
    pdline = [pd[i]*scales[j] for j in range(0, num_scales)]
    qdline = [qd[i]*scales[j] for j in range(0, num_scales)]

    pdout.write('\t'.join(format(x, '10.8f') for x in pdline))
    qdout.write('\t'.join(format(x, '10.8f') for x in qdline))
    pdout.write('\n')
    qdout.write('\n')
pdout.close()
qdout.close()
