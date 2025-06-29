from qdrant_client import QdrantClient, models
from fastembed import TextEmbedding
import numpy as np

model_handle = 'jinaai/jina-embeddings-v2-small-en'


#%% q1
query = 'I just discovered the course. Can I join now?'
embedding_model = TextEmbedding(model_handle)

embedding = next(embedding_model.embed(query)) #type: ignore
print(min(embedding))

#%% q2



