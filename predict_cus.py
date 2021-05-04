from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import numpy as np
import json
import os
import gc

### Constants
THRESHOLD = .75

npylist = np.load('embeddings_final.npy' , allow_pickle = True)
model = SentenceTransformer('distiluse-base-multilingual-cased-v2') ## Model will be downloaded Automatically


embeddings = np.vstack(npylist[:,2])
sentences = np.vstack(npylist[:,1])

del npylist
gc.collect()

sentence = ''

while sentence.lower() != 'exit':
    sentence = str(input('Enter the String (ot type Exit) : '))
    embed = model.encode(sentence , normalize_embeddings = True)
    product = np.dot(embeddings , embed)
    max_loc = np.argpartition(product, -5)[-5:]
    max_loc = max_loc[np.argsort(product[max_loc])]
    print(sentences[max_loc] , '\n',product[max_loc])
    # product = np.sort(product)[-10:]
