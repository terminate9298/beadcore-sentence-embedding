import os
import numpy as np
from tqdm import tqdm
import random
import pickle
from sentence_transformers import SentenceTransformer
import pandas as pd
import gc
np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)


print('Loading Embedding Matrix...')
embeddings = np.load('embeddings_final.npy' , allow_pickle = True)
model = SentenceTransformer('distiluse-base-multilingual-cased-v2' , device = 'cuda') ## Model will be downloaded Automatically

MYDIR = ("noise")
CHECK_FOLDER = os.path.isdir(MYDIR)
if not CHECK_FOLDER:
    os.makedirs(MYDIR)


def noise_sentence(sen , factor: float = .10) -> str:
    output = []
    # factor = random.uniform(0.05,.20)
    for i in range(len(sen)):
        if random.uniform(0,1) < factor:
            output.append(sen[random.randrange(len(sen))])
        else:
            output.append(sen[i])
    return "".join(output)

print('Creating Noised Data...')
data = embeddings[:,:2] ## Change size here
step_size = 1000000
starting_index = 0

del embeddings
gc.collect()

for i in range(starting_index , data.shape[0] , step_size ):
    noised_data = []
    for z in range(step_size):
        if (i+z) < data.shape[0]:
            noised_data.append([data[i+z][0] ,noise_sentence(str(data[i+z][1]))])
        else:
            break
    noised_data = np.array(noised_data)

    f = []
    text = noised_data[:,1]
    unique = noised_data[:,0]
    embed = model.encode(text, device = 'cuda' , batch_size = 256 , normalize_embeddings = True)
    print(f'Step {i}')
    for j,k,l in zip(embed , text , unique):
        f.append([l,k,j])
    np.save(f'noise/embed_{i}.npy', np.array(f) , allow_pickle = True)
