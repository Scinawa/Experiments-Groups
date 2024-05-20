import statistics
from matplotlib import pyplot as plt 

import sys
import subprocess
import os
import numpy as np

directory = os.getcwd()
files = os.listdir(directory)

DATASET_FILES = [file for file in files if sys.argv[1] in file]

plot_me = {}

for filename in DATASET_FILES:
    print ("\nProcessing file: {}".format(filename))

    result = subprocess.run("cat {} | grep Test ".format(filename), shell=True, capture_output=True, text=True)

    # Get the output
    text = result.stdout



    lines = text.strip().split('\n')
    try: 
        accuracies = [float(line.split()[-1]) for line in lines]

        average = statistics.mean(accuracies)
        variance = statistics.variance(accuracies)
        print('File: {}'.format(filename))
        print('Average: {:.6f}'.format(average))
        print('Variance: {:.6f}'.format(variance))
        print('Max:{:.6f}'.format(max(accuracies)) )
        print('Len: {}'.format(len(accuracies)))


        plot_me[filename] = (average, variance, max(accuracies))    


        from time import sleep
        sleep(0.01)
    except Exception as e:
        print(e)


print("WOWOW ")
k = 4
values_so_max = [plot_me["corre{}-{}-so.txt".format(i, sys.argv[1])][2] for i in range(2,k) ]
values_mo_max = [plot_me["corre{}-{}-mo.txt".format(i, sys.argv[1])][2] for i in range(2,k) ]

values_so = [plot_me["corre{}-{}-so.txt".format(i, sys.argv[1])][0] for i in range(2,k) ]
values_mo = [plot_me["corre{}-{}-mo.txt".format(i, sys.argv[1])][0] for i in range(2,k) ]

values_so_sdv = [np.sqrt(plot_me["corre{}-{}-so.txt".format(i, sys.argv[1])][1]) for i in range(2,k) ]
values_mo_sdv = [np.sqrt(plot_me["corre{}-{}-mo.txt".format(i, sys.argv[1])][1]) for i in range(2,k) ]

plt.xticks(range(2,k))

plt.axhline(y = plot_me["original-{}.txt".format(sys.argv[1])][0], color = 'm', linestyle = '-', label="Original (avg)")     # non isomorphic graphs set at the beginning of the notebook
plt.axhline(y = plot_me["original-{}.txt".format(sys.argv[1])][2], color = 'm', linestyle = '--', label="Original (max)")     # non isomorphic graphs set at the beginning of the notebook


plt.plot( [i for i in range(2,k)], values_so_max, label='SO (max)', linestyle = '--', color = 'g')
plt.plot( [i for i in range(2,k)], values_mo_max, label='MO (max)', linestyle = '--', color = 'b')

plt.errorbar([i for i in range(2,k)], values_so, yerr=values_so_sdv, ecolor='lightgray', elinewidth=3, capsize=1, label='SO (avg)', linestyle = '-', color = 'g')
plt.errorbar([i for i in range(2,k)], values_mo, yerr=values_mo_sdv, ecolor='lightblue', elinewidth=3, capsize=1, label='MO (avg)', linestyle = '-', color = 'b')


plt.title("Accuracies of different models on {} dataset".format(sys.argv[1]))
    # plt.yscale('log', base=2)  # Set the scale of the y-axis to logarithmic

    # plt.title("Time needed in hrs")
plt.legend(ncol=2, loc='lower left')
plt.savefig('accuracies-{}.png'.format(sys.argv[1]))
plt.figure()

#### SECOND PLOT

values_so_max_PR = [plot_me["corre{}-{}-PR-so.txt".format(i, sys.argv[1])][2] for i in range(2,k) ]
values_mo_max_PR = [plot_me["corre{}-{}-PR-mo.txt".format(i, sys.argv[1])][2] for i in range(2,k) ]
values_so_PR = [plot_me["corre{}-{}-PR-so.txt".format(i, sys.argv[1])][0] for i in range(2,k) ]
values_mo_PR = [plot_me["corre{}-{}-PR-mo.txt".format(i, sys.argv[1])][0] for i in range(2,k) ]
values_so_PR_sdv = [np.sqrt(plot_me["corre{}-{}-PR-so.txt".format(i, sys.argv[1])][1]) for i in range(2,k) ]
values_mo_PR_sdv = [np.sqrt(plot_me["corre{}-{}-PR-mo.txt".format(i, sys.argv[1])][1]) for i in range(2,k) ]

plt.xticks(range(2,k))

plt.axhline(y = plot_me["original-{}.txt".format(sys.argv[1])][0], color = 'm', linestyle = '-', label="Original (avg)")     # non isomorphic graphs set at the beginning of the notebook
plt.axhline(y = plot_me["original-{}.txt".format(sys.argv[1])][2], color = 'm', linestyle = '--', label="Original (max)")     # non isomorphic graphs set at the beginning of the notebook
# plt.axhline(y = plot_me["original-{}-PR.txt".format(sys.argv[1])][0], color = 'black', linestyle = '-', label="Original (avg)")     # non isomorphic graphs set at the beginning of the notebook
# plt.axhline(y = plot_me["original-{}-PR.txt".format(sys.argv[1])][2], color = 'black', linestyle = '--', label="Original (max)")     # non isomorphic graphs set at the beginning of the notebook



plt.plot( [i for i in range(2,k)], values_so_max_PR, label='SO (max) - PR=0.2', linestyle = '--', color = 'g')
plt.plot( [i for i in range(2,k)], values_mo_max_PR, label='MO (max) - PR=0.2', linestyle = '--', color = 'b')

plt.errorbar([i for i in range(2,k)], values_so_PR, yerr=values_so_PR_sdv, ecolor='lightgray', elinewidth=3, capsize=1, label='SO (avg) - PR=0.2', linestyle = '-', color = 'g')
plt.errorbar([i for i in range(2,k)], values_mo_PR, yerr=values_mo_PR_sdv, ecolor='lightblue', elinewidth=3, capsize=1, label='MO (avg) - PR=0.2', linestyle = '-', color = 'b')


plt.title("Accuracies of different models on {} dataset".format(sys.argv[1]))
    # plt.yscale('log', base=2)  # Set the scale of the y-axis to logarithmic

    # plt.title("Time needed in hrs")
plt.legend(ncol=2, loc='lower left')
plt.savefig('accuracies-second-{}.png'.format(sys.argv[1]))
plt.figure()


### THIRD PLOT


values_so_max_PR = [plot_me["corre{}-{}-M1-so.txt".format(i, sys.argv[1])][2] for i in range(2,k) ]
values_mo_max_PR = [plot_me["corre{}-{}-M1-mo.txt".format(i, sys.argv[1])][2] for i in range(2,k) ]
values_so_PR = [plot_me["corre{}-{}-M1-so.txt".format(i, sys.argv[1])][0] for i in range(2,k) ]
values_mo_PR = [plot_me["corre{}-{}-M1-mo.txt".format(i, sys.argv[1])][0] for i in range(2,k) ]
values_so_PR_sdv = [np.sqrt(plot_me["corre{}-{}-M1-so.txt".format(i, sys.argv[1])][1]) for i in range(2,k) ]
values_mo_PR_sdv = [np.sqrt(plot_me["corre{}-{}-M1-mo.txt".format(i, sys.argv[1])][1]) for i in range(2,k) ]

plt.xticks(range(2,k))

plt.axhline(y = plot_me["original-{}.txt".format(sys.argv[1])][0], color = 'm', linestyle = '-', label="Original (avg)")     # non isomorphic graphs set at the beginning of the notebook
plt.axhline(y = plot_me["original-{}.txt".format(sys.argv[1])][2], color = 'm', linestyle = '--', label="Original (max)")     # non isomorphic graphs set at the beginning of the notebook
# plt.axhline(y = plot_me["original-{}-PR.txt".format(sys.argv[1])][0], color = 'black', linestyle = '-', label="Original (avg)")     # non isomorphic graphs set at the beginning of the notebook
# plt.axhline(y = plot_me["original-{}-PR.txt".format(sys.argv[1])][2], color = 'black', linestyle = '--', label="Original (max)")     # non isomorphic graphs set at the beginning of the notebook



plt.plot( [i for i in range(2,k)], values_so_max_PR, label='SO (max) - M1', linestyle = '--', color = 'g')
plt.plot( [i for i in range(2,k)], values_mo_max_PR, label='MO (max) - M1', linestyle = '--', color = 'b')

plt.errorbar([i for i in range(2,k)], values_so_PR, yerr=values_so_PR_sdv, ecolor='lightgray', elinewidth=3, capsize=1, label='SO (avg) - M1', linestyle = '-', color = 'g')
plt.errorbar([i for i in range(2,k)], values_mo_PR, yerr=values_mo_PR_sdv, ecolor='lightblue', elinewidth=3, capsize=1, label='MO (avg) - M1', linestyle = '-', color = 'b')


plt.title("Accuracies of different models on {} dataset".format(sys.argv[1]))
    # plt.yscale('log', base=2)  # Set the scale of the y-axis to logarithmic

    # plt.title("Time needed in hrs")
plt.legend(ncol=2, loc='lower left')
plt.savefig('accuracies-third-{}.png'.format(sys.argv[1]))
plt.figure()
