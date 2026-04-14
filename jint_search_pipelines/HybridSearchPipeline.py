from azure.search.documents.indexes.models import (
    SearchField,
    SearchFieldDataType,
)

from .AzureSearchPipelineBase import AzureSearchPipelineBase


class HybridSearchPipeline(AzureSearchPipelineBase):
    """
    Pipeline for hybrid search (vector + keyword).

    Enables both vector similarity search and traditional keyword search
    by marking text fields as searchable.
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
