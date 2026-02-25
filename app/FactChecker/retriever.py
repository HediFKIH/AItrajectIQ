
import numpy as np

#query_embeddings = model.encode(queries) 


def retrieve(query_embeddings, index, k=3):
    _, indices = index.search(np.array(query_embeddings, dtype=np.float32), k)  
    return indices

