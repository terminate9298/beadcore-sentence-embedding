from annoy import AnnoyIndex
import numpy as np
import json

npylist = np.load('embeddings_final.npy' , allow_pickle = True)
mapper = {}
annoy_index = AnnoyIndex(512, 'dot')

for i in range(len(npylist)):
    annoy_index.add_item(i,npylist[i][2])
    mapper[i] = {
      'index': int(npylist[i][0]),
      'sentence': str(npylist[i][1])
      }

annoy_index.build(10, -1) # 10 trees
annoy_index.save('annoy_index.ann')

with open('mapper.json', 'w' , encoding="utf-8") as jsonfile:
    json.dump(mapper, jsonfile, indent = 4, ensure_ascii=False)
