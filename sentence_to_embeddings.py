############################################################################################################
## Script to Create Multiple Embedding Files of Sentences in Column 2 of File elastic_mysql_questions_new ##
############################################################################################################

import numpy as np
import pickle
from sentence_transformers import SentenceTransformer
from torch.cuda import is_available
import os
import pandas as pd
from tqdm import tqdm

np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)
print(f'Is Cuda Available {is_available()}')

file_path = 'elastic_mysql_questions_new.csv'
fractio_to_use = 1.0
step_size = 10000 ## Change Step Size
starting_index = 0


MYDIR = ("embeddings")
CHECK_FOLDER = os.path.isdir(MYDIR)
if not CHECK_FOLDER:
    print('Creating Embedding Directory')
    os.makedirs(MYDIR)

print('Loading Sentence Transformer Model')
model = SentenceTransformer('distiluse-base-multilingual-cased-v2' , device = 'cuda') ## Model will be downloaded Automatically

print('Reading CSV File')
questions = pd.read_csv(file_path , header = None) ## Path to the File (Currently in same Directory)

print(f'Original Shape of Dataset {questions.shape[0]}')
questions.drop_duplicates(subset = [2] , inplace = True)
print(f'New Shape of Dataset {questions.shape[0]}')

data = questions.sample(frac = fractio_to_use)
data[2] = data[2].apply(lambda x : str(x))

print('Preparing Embedding of Sentences')
for i in range(starting_index , data.shape[0] , step_size ):
    f = []
    text = data.iloc[i : i+step_size][2].values.reshape(-1)
    unique = data.iloc[i : i+step_size][1].values.reshape(-1)
    embeddings = model.encode(text, device = 'cuda' , batch_size = 256 , normalize_embeddings = True)
    print(f'Step {i}')
    for j,k,l in zip(embeddings , text , unique):
        f.append([l,k,j])
    np.save(f'embeddings/embed_{i}.npy', np.array(f) , allow_pickle = True)
