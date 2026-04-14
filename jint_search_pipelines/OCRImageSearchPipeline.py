import os
from typing import Optional

from azure.search.documents.indexes.models import (
    AzureOpenAIEmbeddingSkill,
    IndexingParameters,
    IndexProjectionMode,
    InputFieldMappingEntry,
    MergeSkill,
    OcrSkill,
    OutputFieldMappingEntry,
    SearchField,
    SearchFieldDataType,
    SearchIndexer,
    SearchIndexerIndexProjection,
    SearchIndexerIndexProjectionSelector,
    SearchIndexerIndexProjectionsParameters,
    SearchIndexerSkillset,
    SplitSkill,
)

from .AzureSearchPipelineBase import AzureSearchPipelineBase, DataSourceType


class OCRImageSearchPipeline(AzureSearchPipelineBase):
    """
    Pipeline for processing PNG images (slides, scanned documents) with OCR.

    Uses Azure's OCR (Optical Character Recognition) skill to extract text from
    images, then processes the extracted text the same way as PDF content.
    Suitable for images of slides, scanned PDFs, or any document images.
    """

    def __init__(
        self,
        index_name: str,
        skillset_name: str,
        azure_search_service: Optional[str] = None,
        azure_openai_account: Optional[str] = None,
        azure_storage_connection: Optional[str] = None,
        azure_ai_services_key: Optional[str] = None,
        embedding_deployment: str = "text-embedding-3-large",
        embedding_model: str = "text-embedding-3-large",
        embedding_dimensions: int = 3072,
        chunk_size: int = 2000,
        chunk_overlap: int = 500,
        sharepoint_app_id: Optional[str] = None,
        sharepoint_app_secret: Optional[str] = None,
        sharepoint_tenant_id: Optional[str] = None,
    ):
        """
        Initialize the OCR Image Search Pipeline.

        Args:
            index_name: Name of the search index to create
            skillset_name: Name of the skillset to create
            azure_search_service: Azure Search service endpoint (defaults to env var)
            azure_openai_account: Azure OpenAI account endpoint (defaults to env var)
            azure_storage_connection: Azure Storage connection string (defaults to env var)
            azure_ai_services_key: Azure AI Services key for OCR (defaults to env var)
            embedding_deployment: Embedding deployment name
            embedding_model: Embedding model name
            embedding_dimensions: Embedding vector dimensions
            chunk_size: Maximum page length for text splitting
            chunk_overlap: Overlap length between chunks
            sharepoint_app_id: SharePoint indexer app registration Client ID
            sharepoint_app_secret: SharePoint indexer app registration secret
            sharepoint_tenant_id: Target Tenant ID for SharePoint
        """
        super().__init__(
            index_name=index_name,
            skillset_name=skillset_name,
            azure_search_service=azure_search_service,
            azure_openai_account=azure_openai_account,
            azure_storage_connection=azure_storage_connection,
            embedding_deployment=embedding_deployment,
            embedding_model=embedding_model,
            embedding_dimensions=embedding_dimensions,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            sharepoint_app_id=sharepoint_app_id,
            sharepoint_app_secret=sharepoint_app_secret,
            sharepoint_tenant_id=sharepoint_tenant_id,
        )
        # OCR configuration
        self.azure_ai_services_key = azure_ai_services_key or os.environ.get(
            "AZURE_AI_SERVICES_KEY"
        )

    def _create_fields(self) -> list[SearchField]:
        """
        Create fields with searchable text fields for image search.

        Returns:
            List of SearchField objects with title and chunk marked as searchable
        """
        return [
            SearchField(name="parent_id", type=SearchFieldDataType.String),
            SearchField(
                name="title",
                type=SearchFieldDataType.String,
                searchable=True,
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
                searchable=True,
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

    def create_skillset(
        self, data_source_type: DataSourceType = DataSourceType.AZURE_BLOB
    ) -> SearchIndexerSkillset:
        """
        Create the skillset for image processing with OCR.

        The skillset includes:
        - OCR skill to extract text from images (PNG, JPEG, etc.)
        - Text splitting into chunks
        - Embedding generation via Azure OpenAI
        - Index projection for chunk-level indexing

        Args:
            data_source_type: Type of data source to determine URL field mapping

        Returns:
            Created SearchIndexerSkillset object
        """
        if not self.index:
            raise ValueError(
                "Index must be created before skillset. Call create_index() first."
            )

        # OCR skill to extract text from images
        ocr_skill = OcrSkill(
            description="Extract text from images using OCR",
            context="/document/normalized_images/*",
            inputs=[
                InputFieldMappingEntry(
                    name="image", source="/document/normalized_images/*"
                ),
            ],
            outputs=[OutputFieldMappingEntry(name="text", target_name="text")],
        )

        # Merge skill to combine document content with OCR text
        merge_skill = MergeSkill(
            description="Merge OCR text with document content",
            context="/document",
            inputs=[
                InputFieldMappingEntry(name="text", source="/document/content"),
                InputFieldMappingEntry(
                    name="itemsToInsert", source="/document/normalized_images/*/text"
                ),
                InputFieldMappingEntry(
                    name="offsets", source="/document/normalized_images/*/contentOffset"
                ),
            ],
            outputs=[
                OutputFieldMappingEntry(name="mergedText", target_name="merged_text")
            ],
        )

        # Split skill to chunk the merged text
        split_skill = SplitSkill(
            description="Split merged text into chunks",
            text_split_mode="pages",
            context="/document",
            maximum_page_length=self.chunk_size,
            page_overlap_length=self.chunk_overlap,
            inputs=[
                InputFieldMappingEntry(name="text", source="/document/merged_text"),
            ],
            outputs=[OutputFieldMappingEntry(name="textItems", target_name="pages")],
        )

        # Embedding skill to generate vectors
        embedding_skill = AzureOpenAIEmbeddingSkill(
            description="Skill to generate embeddings via Azure OpenAI",
            context="/document/pages/*",
            resource_url=self.azure_openai_account,
            deployment_name=self.embedding_deployment,
            model_name=self.embedding_model,
            dimensions=self.embedding_dimensions,
            inputs=[
                InputFieldMappingEntry(name="text", source="/document/pages/*"),
            ],
            outputs=[
                OutputFieldMappingEntry(name="embedding", target_name="text_vector")
            ],
        )

        # Get the appropriate URL source and title source based on data source type
        file_url_source = self._get_file_url_source(data_source_type)
        title_source = self._get_title_source(data_source_type)

        # Index projection for chunk-level indexing
        index_projections = SearchIndexerIndexProjection(
            selectors=[
                SearchIndexerIndexProjectionSelector(
                    target_index_name=self.index.name,
                    parent_key_field_name="parent_id",
                    source_context="/document/pages/*",
                    mappings=[
                        InputFieldMappingEntry(
                            name="chunk", source="/document/pages/*"
                        ),
                        InputFieldMappingEntry(
                            name="text_vector", source="/document/pages/*/text_vector"
                        ),
                        InputFieldMappingEntry(name="title", source=title_source),
                        InputFieldMappingEntry(name="file_url", source=file_url_source),
                    ],
                ),
            ],
            parameters=SearchIndexerIndexProjectionsParameters(
                projection_mode=IndexProjectionMode.SKIP_INDEXING_PARENT_DOCUMENTS
            ),
        )

        # Create skillset with OCR, merge, split, and embedding skills
        skillset = SearchIndexerSkillset(
            name=self.skillset_name,
            description="Skillset to process images with OCR, merge text, chunk, and generate embeddings",
            skills=[ocr_skill, merge_skill, split_skill, embedding_skill],
            index_projection=index_projections,
        )

        result = self.indexer_client.create_or_update_skillset(skillset)
        self.skillset = result
        print(f"✓ Skillset '{result.name}' created successfully with OCR support")

        return result

    def create_indexer(
        self,
        indexer_name: str,
        data_source_name: str,
        batch_size: int = 20,
        max_failed_items: int = -1,
    ) -> SearchIndexer:
        """
        Create an indexer to process images from the data source with OCR.

        This override configures the indexer to extract and normalize images
        before passing them to the OCR skill.

        Args:
            indexer_name: Name for the indexer
            data_source_name: Name of the data source to use
            batch_size: Number of documents to process in each batch
            max_failed_items: Maximum failed items before stopping (-1 for unlimited)

        Returns:
            Created SearchIndexer object
        """
        if not self.index or not self.skillset:
            raise ValueError(
                "Index and skillset must be created before indexer. "
                "Call create_index() and create_skillset() first."
            )

        indexer_parameters = IndexingParameters(
            batch_size=batch_size,
            max_failed_items_per_batch=max_failed_items,
            configuration={"imageAction": "generateNormalizedImages"},
        )

        indexer = SearchIndexer(
            name=indexer_name,
            description="Indexer to process images with OCR and generate embeddings",
            skillset_name=self.skillset.name,
            target_index_name=self.index.name,
            data_source_name=data_source_name,
            parameters=indexer_parameters,
        )

        result = self.indexer_client.create_or_update_indexer(indexer)
        print(
            f"✓ Indexer '{indexer_name}' created successfully and is running. "
            f"Give the indexer a few minutes to process images with OCR."
        )

        return result
