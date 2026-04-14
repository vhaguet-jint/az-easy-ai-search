"""
Example: RAG chat using an Azure AI Search index.

Set the constants below and run:
    python examples/rag_chat.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from azure.identity import DefaultAzureCredential, get_bearer_token_provider
from azure.search.documents import SearchClient
from azure.search.documents.models import VectorizableTextQuery
from dotenv import load_dotenv
from openai import AzureOpenAI

from prompts import GROUNDED_PROMPT

load_dotenv()

# ── configure these ──────────────────────────────────────────────────────────
INDEX_NAME = "my-index"
DEPLOYMENT_NAME = "gpt-4o"  # your Azure OpenAI chat deployment
# ─────────────────────────────────────────────────────────────────────────────

AZURE_SEARCH_SERVICE: str = os.environ["AZURE_SEARCH_SERVICE"]
AZURE_OPENAI_ACCOUNT: str = os.environ["AZURE_OPENAI_ACCOUNT"]

credential = DefaultAzureCredential()
token_provider = get_bearer_token_provider(
    credential, "https://cognitiveservices.azure.com/.default"
)
openai_client = AzureOpenAI(
    api_version="2024-06-01",
    azure_endpoint=AZURE_OPENAI_ACCOUNT,
    azure_ad_token_provider=token_provider,
)


def ask_question(query: str, search_client: SearchClient) -> str:
    vector_query = VectorizableTextQuery(
        text=query, k_nearest_neighbors=50, fields="text_vector"
    )

    search_results = search_client.search(
        query_type="semantic",
        search_text=query,
        vector_queries=[vector_query],
        select=["title", "chunk"],
        top=5,
        semantic_configuration_name="semantic-config",
    )

    documents = list(search_results)

    for i, doc in enumerate(documents):
        print(f"  [{i + 1}] {doc['title']}")

    sources_formatted = "=================\n".join(
        [f"TITLE: {doc['title']}, CONTENT: {doc['chunk']}" for doc in documents]
    )

    response = openai_client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": GROUNDED_PROMPT.format(
                    query=query, sources=sources_formatted
                ),
            }
        ],
        model=DEPLOYMENT_NAME,
    )

    return response.choices[0].message.content


if __name__ == "__main__":
    query = input("Question: ")
    search_client = SearchClient(
        endpoint=AZURE_SEARCH_SERVICE, credential=credential, index_name=INDEX_NAME
    )
    answer = ask_question(query=query, search_client=search_client)
    print("\nAnswer:", answer)
