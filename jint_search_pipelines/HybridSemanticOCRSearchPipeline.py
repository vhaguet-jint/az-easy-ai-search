from azure.search.documents.indexes.models import (
    SemanticConfiguration,
    SemanticField,
    SemanticPrioritizedFields,
    SemanticSearch,
)
from dotenv import load_dotenv

from .OCRImageSearchPipeline import OCRImageSearchPipeline

load_dotenv()


class HybridSemanticOCRSearchPipeline(OCRImageSearchPipeline):
    """
    Pipeline for image search with OCR + semantic reranking.

    Combines OCR text extraction, vector similarity, keyword search, and
    semantic reranking for the most advanced image search capabilities.
    """

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
