###############################################################
## Script to Merge Multiple Files created by previous Script ##
###############################################################

import numpy as np
import os

filelist = next(os.walk('embeddings/'))[2]
npylist = [np.load('embeddings/' + e, allow_pickle = True).tolist() for e in filelist]
npylist = np.vstack(npylist)

np.save('embeddings_final.npy' , npylist)
