import matplotlib.pyplot as plt 
import numpy as np

fig = plt.figure(1)

# Expert
plt.errorbar(np.arange(0, 11, 1), [5516.52 for i in range(11)], yerr=[58.31 for i in range(11)], label='Expert')

# Behavioral Cloning
plt.errorbar(np.arange(0, 11, 1), [2522.23 for i in range(11)], yerr=[968.91 for i in range(11)],
             label='Behavioral Cloning')

# DAgger
plt.errorbar(np.arange(0, 11, 1), [2522, 2779, 5192, 5531, 5501, 5389, 5467, 5529, 5467, 5576, 5516],
             yerr=[968.9, 1945, 610.7, 32.37, 94.65, 130.5, 65.88, 40.18, 69.29, 54.57, 55.95], fmt='-o',
             label='DAgger')

plt.title('Return Mean over 20 Rollouts')
plt.legend()
plt.xlabel('times of data augmentation')
plt.ylabel('performance after the corresponding augmentation')

plt.show()
