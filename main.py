import json
from openai import OpenAI
from elasticsearch import Elasticsearch
from tqdm import tqdm

client = OpenAI()
es_client = Elasticsearch("http://localhost:9200")

with open("documents.json", "r") as f:
    docs_raw = json.load(f)

documents = []

for course_dict in docs_raw:
    for doc in course_dict["documents"]:
        doc["course"] = course_dict["course"]
        documents.append(doc)


index_settings = {
    "settings": {"number_of_shards": 1, "number_of_replicas": 0},
    "mappings": {
        "properties": {
            "text": {"type": "text"},
            "section": {"type": "text"},
            "question": {"type": "text"},
            "course": {"type": "keyword"},
        }
    },
}
index_name = "course_questions"
es_client.indices.create(
    index=index_name,
    body=index_settings
)

for doc in tqdm(documents):
    es_client.index(index= index_name,document= doc)

def elastic_search(query):

    search_query = {
        "size": 3,
        "query": {
            "bool": {
                "must": {
                    "multi_match": {
                        "query": query,
                        "fields": ["question^4", "text"], #^X means X times more important
                        "type": "best_fields"
                    }
                },
                "filter": {
                    "term": {
                        "course": "machine-learning-zoomcamp"
                    }
                }
            }
        }
    }

    response = es_client.search(index=index_name, body=search_query )
    result_docs = []
    for hit in response['hits']['hits']:
        result_docs.append(hit['_source'])

    return result_docs


def build_prompt(query, search_results):
    context_template = """
    Q: {question}
    A: {text}
    """.strip()

    context = ""

    for doc in search_results:
        context = (
            context
            + '\n\n'
            + context_template.format(question=doc['question'],text=doc['text']) 
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
    response = client.chat.completions.create(
        model="gpt-4o-mini", messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content


def rag(query):
    search_results = elastic_search(query)
    prompt = build_prompt(query, search_results)
    answer = llm(prompt)
    return answer


query = "How do copy a file to a Docker container?"
print(rag(query))
