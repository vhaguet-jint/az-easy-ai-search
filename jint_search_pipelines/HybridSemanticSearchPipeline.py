from azure.search.documents.indexes.models import (
    SearchField,
    SearchFieldDataType,
    SemanticConfiguration,
    SemanticField,
    SemanticPrioritizedFields,
    SemanticSearch,
)
from dotenv import load_dotenv

from .AzureSearchPipelineBase import AzureSearchPipelineBase

load_dotenv()


class HybridSemanticSearchPipeline(AzureSearchPipelineBase):
    """
    Pipeline for hybrid search with semantic reranking.

    Combines vector similarity, keyword search, and semantic reranking
    for the most advanced search capabilities.
    """

    def _create_fields(self) -> list[SearchField]:
        """
        Create fields with searchable text fields for hybrid search.

        Returns:
            List of SearchField objects with title and chunk marked as searchable
        """
        return [
            SearchField(name="parent_id", type=SearchFieldDataType.String),
            SearchField(
                name="title",
                type=SearchFieldDataType.String,
                searchable=True,  # Enable keyword search on title
            ),
            SearchField(
                name="chunk_id",
                type=SearchFieldDataType.String,
                key=True,
                sortable=True,
                filterable=True,
                facetable=True,
                analyzer_name="keyword",
            ),
            SearchField(
                name="chunk",
                type=SearchFieldDataType.String,
                searchable=True,  # Enable keyword search on content
                sortable=False,
                filterable=False,
                facetable=False,
            ),
            SearchField(
                name="text_vector",
                type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
                vector_search_dimensions=self.embedding_dimensions,
                vector_search_profile_name="myHnswProfile",
            ),
            SearchField(
                name="file_url",
                type=SearchFieldDataType.String,
                searchable=False,
                sortable=True,
                filterable=True,
            ),
        ]

    def _create_semantic_search(self) -> SemanticSearch:
        """
        Create semantic search configuration for reranking.

        Returns:
            SemanticSearch configuration object
        """
        semantic_config = SemanticConfiguration(
            name="semantic-config",
            prioritized_fields=SemanticPrioritizedFields(
                title_field=SemanticField(field_name="title"),
                content_fields=[SemanticField(field_name="chunk")],
                keywords_fields=[SemanticField(field_name="chunk")],
            ),
        )
        return SemanticSearch(configurations=[semantic_config])
