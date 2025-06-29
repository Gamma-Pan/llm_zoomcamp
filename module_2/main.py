from qdrant_client import QdrantClient, models
import requests
import openai
import random
import json


# %% clients
openai_client = openai.Client()

client = QdrantClient("http://localhost:6333")  # connect to local Qdrant instance

# %% download data
docs_url = "https://github.com/alexeygrigorev/llm-rag-workshop/raw/main/notebooks/documents.json"
docs_response = requests.get(docs_url)
documents_raw = docs_response.json()

# %% create collection in qdrant
client.create_collection(
    collection_name="zoomcamp-rag",
    vectors_config=models.VectorParams(size=1536, distance=models.Distance.COSINE),
)

# %% emdbed text data using openai
docs = []

for course in documents_raw:
    for doc in course["documents"]:
        doc["course"] = course["course"]
        docs.append(doc)

vectors = openai_client.embeddings.create(
    input=[x["text"] for x in docs], model="text-embedding-3-small"
)

points = [
    models.PointStruct(
        id=idx,
        vector=data.embedding,
        payload={
            "text": doc["text"],
            "section": doc["section"],
            "course": doc["course"],
        },
    )
    for idx, (data, doc) in enumerate(zip(vectors.data, docs))
]

client.upsert("zoomcamp-rag", points)

# %% search qdrant
client.create_payload_index(
    collection_name="zoomcamp-rag",
    field_name="course",
    field_schema="keyword", #type: ignore
)


def search(query, course="mlops-zoomcamp", limit=1):
    query = openai_client.embeddings.create(input=query, model="text-embedding-3-small")
    results = client.query_points(
        collection_name="zoomcamp-rag",
        query=query.data[0].embedding,
        query_filter=models.Filter(
            must=[
                models.FieldCondition(
                    key="course", match=models.MatchValue(value=course)
                )
            ]
        ),
        limit=limit,
        with_payload=True,
    )
    return results


# pick a random questio from the course data
course = random.choice(documents_raw)
course_piece = random.choice(course["documents"])
print(json.dumps(course_piece, indent=2))


# search_res = search(course_piece['question'])
# print(search_res)

#print(search('what if i submit homeworks late?', "mlops-zoomcamp").points[0].payload['text'])
213
