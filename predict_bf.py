from tqdm import tqdm
import numpy as np
import json
import os
import gc

### Constants
THRESHOLD = .75

npylist = np.load('embeddings_final.npy' , allow_pickle = True)
noised_data = np.load('noise/embed_0.npy' , allow_pickle = True)

embeddings = np.vstack(npylist[:,2])
sentences = np.vstack(npylist[:,1])

n_embeddings = np.vstack(noised_data[:,2])
n_sentences = np.vstack(noised_data[:,1])

del npylist , noised_data
gc.collect()

count = 0
acc = 0
wrong = 0
top_5 = 0

for i in range(len(n_embeddings)):
    count+=1
    product = np.dot(embeddings , n_embeddings[i])

    top_loc = np.argmax(product)
    top_pro = product[top_loc]

    if top_pro > THRESHOLD:
        if top_loc == i:
            acc += 1
        else:
            original_d = np.dot(n_embeddings[i] , embeddings[i])
            # if original_d > top_pro:
            if True:
                print(f"\
                        Noised Sentence ---> {n_sentences[i]}\n \
                        Actual Sentence ---> {sentences[i]}\n \
                        Dot distance (Noised and Actual)---> {original_d}\n \
                        Predicted Sentence ---> {sentences[top_loc]}\n \
                        Dot distance (Noised and Predicted)---> {top_pro}\n \
                        Value of Result ---> {top_loc} , {i} \n\
                        ")
            wrong += 1
    if count % 1000 == 0:
        td = round((acc/count)*100,1)
        fa = round((wrong/count)*100,1)
        print(f'Total : {count} , Correct : {acc} Wrong : {wrong}   -->> TD - {td}%  / FA - {fa}%')
