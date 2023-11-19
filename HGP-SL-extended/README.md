#On my machine
tar -zcvf PROTEINS_matrices.tar.gz PROTEINS/
scp PROTEINS_matrices.tar.gz nus8:/hpctmp/cqtales/datasets/



On the HPC machine:
# 1
tar -zxvf PROTEINS_skew.tar.gz /hpctmp/cqtales/datasets/PROTEINS/

# 2
# Compute the MORSKSP with the script psb-script

# 3
# Create new compressed archive of morkssp
tar -zcvf /hpctmp/cqtales/datasets/PROTEINS_morsksp.tar.gz /hpctmp/cqtales/datasets/PROTEINS/

# 4
Copy on machine
scp nus8:/hpctmp/cqtales/datasets/PROTEINS_morsksp.tar.gz .

# 5
tar zcvf PROTEINS_morsksp.tar.gz PROTEINS/



