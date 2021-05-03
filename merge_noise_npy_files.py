###############################################################
## Script to Merge Multiple Files created by previous Script ##
###############################################################

import numpy as np
import os
np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)

filelist = next(os.walk('noise/'))[2]
npylist = [np.load('noise/' + e, allow_pickle = True).tolist() for e in filelist]
npylist = np.vstack(npylist)

np.save('noised_data.npy' , npylist)
