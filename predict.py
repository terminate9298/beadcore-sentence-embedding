import numpy as np
from annoy import AnnoyIndex
from tqdm import tqdm
import json
import os

annoy_index = AnnoyIndex(512, 'dot')
annoy_index.load('annoy_index.ann')

noised_data = np.load('noise/embed_0.npy' , allow_pickle = True)
npylist = np.load('embeddings_final.npy' , allow_pickle = True)


threshold = .5 #.9
count = 0
acc = 0
wrong = 0


for i in range(len(noised_data)):
    count += 1
    value = noised_data[i][2]
    result , distance = annoy_index.get_nns_by_vector(value , 5 , include_distances= True)

    if distance[0] >= threshold:
        if result[0] == i:
            acc += 1
        else:
            vector = npylist[i][2]
            d_vector = np.dot(vector , value)
            vector = npylist[result[0]][2]
            p_vector = np.dot(vector ,value)
            if p_vector < d_vector:
                print(f"\
                        Noised Sentence ---> {noised_data[i][1]}\n \
                        Actual Sentence ---> {npylist[i][1]}\n \
                        Dot distance (Noised and Actual)---> {d_vector}\n \
                        Predicted Sentence ---> {npylist[result[0]][1]}\n \
                        Dot distance (Noised and Predicted)---> {p_vector}\n \
                        Value of Result ---> {result[0]} , {i} \n\
                        ")

            wrong += 1
    if count% 1000 == 0:
        td = round((acc/count)*100,1)
        fa = round((wrong/count)*100,1)
        print(f'Total : {count} , Correct : {acc} Wrong : {wrong}   -->> TD - {td}%  / FA - {fa}%')
