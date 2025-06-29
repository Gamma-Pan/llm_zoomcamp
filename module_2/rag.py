import json
from openai import OpenAI
from qdrant_client import QdrantClient, models

openai_client = OpenAI()
qd_client = QdrantClient("http://localhost:6333")
EMBEDDING_DIM = 1536

collection_name = "zoomcamp-faq"
qd_client.create_collection(
    collection_name=collection_name,
    vectors_config=models.VectorParams(
        size=EMBEDDING_DIM, distance=models.Distance.COSINE
    ),
)

qd_client.create_payload_index(
    collection_name="zoomcamp-rag",
    field_name="course",
    field_schema="keyword", #type: ignore
)

with open("documents.json", "r") as f:
    docs_raw = json.load(f)

documents = []

for course_dict in docs_raw:
    for doc in course_dict["documents"]:
        doc["course"] = course_dict["course"]
        documents.append(doc)


embed_responce = openai_client.embeddings.create(
    input=[x["question"] + " " + x["text"] for x in documents],
    model="text-embedding-3-small",
)

points = [
    models.PointStruct(
        id=idx,
        vector=data.embedding,
        payload=doc,
    )
    for idx, (data, doc) in enumerate(zip(embed_responce.data, documents))
]

qd_client.upsert("zoomcamp-faq", points)

def vector_search(question):
    embed_responce = openai_client.embeddings.create(
        input=question, model="text-embedding-3-small"
    )
    course="data-engineering-zoomcamp"
    query_points = qd_client.query_points(
        collection_name=collection_name,
        query=embed_responce.data[0].embedding,
        query_filter=models.Filter(
                must=[
                    models.FieldCondition(
                        key="course", match=models.MatchValue(value=course)
                    )
                ]
            ),
        limit=5,
        with_payload=True,
    )
    results = [point.payload for point in query_points.points]
    return results

def build_prompt(query, search_results):
    context_template = """
    Q: {question}
    A: {text}
    """.strip()

    context = ""

    for doc in search_results:
        context = (
            context
            + "\n\n"
            + context_template.format(question=doc["question"], text=doc["text"])
        )

    prompt_template = """
    You're a course teaching assistant. Answer the QUESTION based on the CONTEXT from the FAQ database.
    Use only the facts from the CONTEXT when answering the QUESTION.

    QUESTION: {question}

    CONTEXT:
    {context}
    """.strip()

    prompt = prompt_template.format(question=query, context=context).strip()

    return prompt


def llm(prompt):
    response = openai_client.chat.completions.create(
        model="gpt-4o-mini", messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content


def rag(query):
    search_results = vector_search(query)
    prompt = build_prompt(query, search_results)
    answer = llm(prompt)
    return answer


query = "How do I run Kafka?"
ans = rag(query)
