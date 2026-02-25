from unicodedata import category
from fastapi import APIRouter, HTTPException, Request
from typing import List
from .retriever import retrieve
import time
from resources.models.category_mapper import Category
from resources.models.doc_store import Chunk
import faiss
import numpy as np


router = APIRouter(
    prefix="/retrieve",
    tags=["retriever"]
)






@router.post("/")
async def retrieve_contexts(request:Request, queries: List[str]):
    #with open("results.txt", "w", encoding="utf-8") as f:
    start = time.perf_counter()
    category_index = request.app.state.category_index
    model= request.app.state.embedding_model
    print(type(queries))
    query_embeddings=model.encode(queries,batch_size=8,show_progress_bar=True)
    category_indices=retrieve(query_embeddings, category_index)
    end = time.perf_counter()
    #f.write(f"Category retrieval Took {end - start:.6f} seconds \n")
    print(f"Category retrieval Took {end - start:.6f} seconds")
    start = time.perf_counter()
    categories=[]
    for query_categories in category_indices:
        query_categories= query_categories.tolist()
        categories.append(await Category.find({"category_id": {"$in": query_categories}}).to_list() )

    end = time.perf_counter()
    # f.write(f"Category context retreival Took {end - start:.6f} seconds \n")
    print(f"Category context retreival Took {end - start:.6f} seconds")
    # for i, query_categories in enumerate(categories):
    #     f.write(f"query:  {queries[i]} \n")
    #     f.write(f"categories: \n")
        
    #     for category in query_categories:
    #         f.write(f"{category.category} \n")


    start = time.perf_counter()

    target_ids=[]
    for query_categories in categories: 
        ids=[]
        for category in query_categories:
            ids+= category.chunk_ids
        ids = sorted(set(ids))
        target_ids.append(ids)
    target_ids_arrays = [np.array(ids, dtype=np.int64) for ids in target_ids]
    selectors = [
        faiss.IDSelectorBatch(len(arr), faiss.swig_ptr(arr))
        for arr in target_ids_arrays
    ]

    index=request.app.state.faiss_index


    results = []
    for query ,selector in zip(query_embeddings, selectors):
        params = faiss.SearchParameters()
        params.sel = selector
        _, indices = index.search(query.reshape(1, -1), 3, params=params)
        results.append(indices)



    end = time.perf_counter()
    print(f"Chunk retreival Took {end - start:.6f} seconds")
    # f.write(f"Chunk retreival Took {end - start:.6f} seconds\n")
    start = time.perf_counter()




    chunks=[]
    for query_chunks_indices in results:

        query_chunks_indices= query_chunks_indices.tolist()[0]
        chunks.append(await Chunk.find({"chunk_id": {"$in": query_chunks_indices}}).to_list() )


    end = time.perf_counter()
    print(f"Chunk context retreival Took {end - start:.6f} seconds")
    # f.write(f"Chunk context retreival Took {end - start:.6f} seconds \n")

    res=[]

    for i, query_chunks in enumerate(chunks):
        # f.write(f"query:  {queries[i]} \n")
        # f.write(f"retrieved contexts: \n")
        contexts=[]

        for chunk in query_chunks:
            contexts.append(chunk.text)
            # f.write(f"{chunk.text}\n")
        res.append(contexts)

    return res






