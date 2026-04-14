"""
Azure AI Search Pipeline Classes

This module provides a class hierarchy for creating Azure AI Search indexes with different
search capabilities:

For PDF Documents:
- VectorSearchPipeline: Basic vector search
- HybridSearchPipeline: Vector + keyword search
- HybridSemanticSearchPipeline: Vector + keyword + semantic reranking

For PNG/Image Documents (with OCR):
- OCRImageSearchPipeline: OCR text extraction + vector search (for PNG, JPEG, images of slides/PDFs)
- HybridSemanticOCRSearchPipeline: OCR + vector + keyword + semantic reranking

Supports data sources:
- Azure Blob Storage (azureblob): Standard blob containers
- SharePoint Online (sharepoint): SharePoint document libraries with managed identity auth
"""

import os
from abc import ABC
from enum import Enum
from typing import Optional

from azure.identity import DefaultAzureCredential
from azure.search.documents.indexes import SearchIndexClient, SearchIndexerClient
from azure.search.documents.indexes.models import (
    AzureOpenAIEmbeddingSkill,
    AzureOpenAIVectorizer,
    AzureOpenAIVectorizerParameters,
    FieldMapping,
    HnswAlgorithmConfiguration,
    IndexingParameters,
    IndexProjectionMode,
    InputFieldMappingEntry,
    OutputFieldMappingEntry,
    SearchField,
    SearchFieldDataType,
    SearchIndex,
    SearchIndexer,
    SearchIndexerDataContainer,
    SearchIndexerDataSourceConnection,
    SearchIndexerIndexProjection,
    SearchIndexerIndexProjectionSelector,
    SearchIndexerIndexProjectionsParameters,
    SearchIndexerSkillset,
    SemanticSearch,
    SplitSkill,
    VectorSearch,
    VectorSearchProfile,
)


def _require_env(key: str, provided: Optional[str] = None) -> str:
    if provided:
        return provided
    value = os.environ.get(key)
    if not value:
        raise ValueError(
            f"Missing required configuration '{key}'. "
            f"Pass it as a constructor argument or set the {key} environment variable."
        )
    return value


class DataSourceType(Enum):
    """
    Supported data source types for Azure AI Search indexers.

    - AZURE_BLOB: Azure Blob Storage containers
    - SHAREPOINT: SharePoint Online document libraries
    """

    AZURE_BLOB = "azureblob"
    SHAREPOINT = "sharepoint"


class AzureSearchPipelineBase(ABC):
    """
    Base class for Azure AI Search index pipelines.

    Handles common initialization, index creation, skillset creation,
    data source creation, and indexer creation. Child classes can override
    specific methods to customize search behavior.
    """

    def __init__(
        self,
        index_name: str,
        skillset_name: str,
        azure_search_service: Optional[str] = None,
        azure_openai_account: Optional[str] = None,
        azure_storage_connection: Optional[str] = None,
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
        Initialize the Azure Search Pipeline.

        Args:
            index_name: Name of the search index to create
            skillset_name: Name of the skillset to create
            azure_search_service: Azure Search service endpoint (defaults to env var)
            azure_openai_account: Azure OpenAI account endpoint (defaults to env var)
            azure_storage_connection: Azure Storage connection string (defaults to env var)
            embedding_deployment: Embedding deployment name
            embedding_model: Embedding model name
            embedding_dimensions: Embedding vector dimensions
            chunk_size: Maximum page length for text splitting
            chunk_overlap: Overlap length between chunks
            sharepoint_app_id: SharePoint indexer app registration Client ID
            sharepoint_app_secret: SharePoint indexer app registration secret
            sharepoint_tenant_id: Target Tenant ID for SharePoint (where the data resides)
        """
        # Service endpoints and credentials
        self.azure_search_service = _require_env("AZURE_SEARCH_SERVICE", azure_search_service)
        self.azure_openai_account = _require_env("AZURE_OPENAI_ACCOUNT", azure_openai_account)
        self.azure_storage_connection = _require_env("AZURE_STORAGE_CONNECTION", azure_storage_connection)

        # SharePoint integration credentials (optional — only required when using SharePoint data source)
        self.sharepoint_app_id = sharepoint_app_id or os.environ.get("SHAREPOINT_INDEXER_APP_ID")
        self.sharepoint_app_secret = sharepoint_app_secret or os.environ.get("SHAREPOINT_INDEXER_APP_SECRET")
        self.sharepoint_tenant_id = sharepoint_tenant_id or os.environ.get("SHAREPOINT_TENANT_ID")

        # Pipeline configuration
        self.index_name = index_name
        self.skillset_name = skillset_name
        self.embedding_deployment = embedding_deployment
        self.embedding_model = embedding_model
        self.embedding_dimensions = embedding_dimensions
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        # Azure credentials and clients
        self.credential = DefaultAzureCredential()
        self.index_client = SearchIndexClient(
            endpoint=self.azure_search_service, credential=self.credential
        )
        # Use preview API version for SharePoint indexer support
        # SharePoint indexer is a preview feature requiring api-version=2025-11-01-preview
        self.indexer_client = SearchIndexerClient(
            endpoint=self.azure_search_service,
            credential=self.credential,
            api_version="2025-11-01-preview",
        )

        # Storage for created resources
        self.index: Optional[SearchIndex] = None
        self.skillset: Optional[SearchIndexerSkillset] = None

    def _create_fields(self) -> list[SearchField]:
        """
        Create the field schema for the search index.

        Base implementation creates fields for vector search only.
        Override in child classes to enable additional search capabilities.

        Returns:
            List of SearchField objects defining the index schema
        """
        return [
            SearchField(name="parent_id", type=SearchFieldDataType.String),
            SearchField(name="title", type=SearchFieldDataType.String),
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

    def _create_vector_search(self) -> VectorSearch:
        """
        Create the vector search configuration.

        This configuration is the same for all pipeline types.

        Returns:
            VectorSearch configuration object
        """
        return VectorSearch(
            algorithms=[
                HnswAlgorithmConfiguration(name="myHnsw"),
            ],
            profiles=[
                VectorSearchProfile(
                    name="myHnswProfile",
                    algorithm_configuration_name="myHnsw",
                    vectorizer_name="myOpenAI",
                )
            ],
            vectorizers=[
                AzureOpenAIVectorizer(
                    vectorizer_name="myOpenAI",
                    kind="azureOpenAI",
                    parameters=AzureOpenAIVectorizerParameters(
                        resource_url=self.azure_openai_account,
                        deployment_name=self.embedding_deployment,
                        model_name=self.embedding_model,
                    ),
                ),
            ],
        )

    def _create_semantic_search(self) -> Optional[SemanticSearch]:
        """
        Create the semantic search configuration.

        Base implementation returns None (no semantic search).
        Override in child classes to enable semantic reranking.

        Returns:
            SemanticSearch configuration object or None
        """
        return None

    def create_index(self) -> SearchIndex:
        """
        Create the search index with the configured schema and search capabilities.

        Returns:
            Created SearchIndex object
        """
        fields = self._create_fields()
        vector_search = self._create_vector_search()
        semantic_search = self._create_semantic_search()

        # Build index with optional semantic search
        index = SearchIndex(
            name=self.index_name,
            fields=fields,
            vector_search=vector_search,
            semantic_search=semantic_search,
        )

        result = self.index_client.create_or_update_index(index)
        self.index = result
        print(f"✓ Index '{result.name}' created successfully")

        return result

    def _get_file_url_source(self, data_source_type: DataSourceType) -> str:
        """
        Get the appropriate file URL source field based on data source type.

        Args:
            data_source_type: The type of data source (AZURE_BLOB or SHAREPOINT)

        Returns:
            The source path for file URL metadata
        """
        if data_source_type == DataSourceType.SHAREPOINT:
            # SharePoint provides the full URL in metadata_spo_item_weburl
            return "/document/metadata_spo_item_weburi"
        else:
            # Blob storage - use storage path. URL will need to be constructed client-side
            # Format: https://{storage_account}.blob.core.windows.net/{container}/{blob_name}
            # Format: https://{storage_account}.blob.core.windows.net/{container}/{blob_name}
            return "/document/metadata_storage_path"

    def _get_title_source(self, data_source_type: DataSourceType) -> str:
        """
        Get the appropriate title source field based on data source type.

        Args:
            data_source_type: The type of data source (AZURE_BLOB or SHAREPOINT)

        Returns:
            The source path for title metadata
        """
        if data_source_type == DataSourceType.SHAREPOINT:
            return "/document/metadata_spo_item_name"
        else:
            return "/document/metadata_storage_name"

    def create_skillset(
        self, data_source_type: DataSourceType = DataSourceType.AZURE_BLOB
    ) -> SearchIndexerSkillset:
        """
        Create the skillset for document processing.

        The skillset includes:
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

        # Split skill to chunk documents
        split_skill = SplitSkill(
            description="Split skill to chunk documents",
            text_split_mode="pages",
            context="/document",
            maximum_page_length=self.chunk_size,
            page_overlap_length=self.chunk_overlap,
            inputs=[
                InputFieldMappingEntry(name="text", source="/document/content"),
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

        # Create skillset
        skillset = SearchIndexerSkillset(
            name=self.skillset_name,
            description="Skillset to chunk documents and generate embeddings",
            skills=[split_skill, embedding_skill],
            index_projection=index_projections,
        )

        result = self.indexer_client.create_or_update_skillset(skillset)
        self.skillset = result
        print(f"✓ Skillset '{result.name}' created successfully")

        return result

    def create_data_source(
        self,
        data_source_name: str,
        container_name: str,
        data_source_type: DataSourceType = DataSourceType.AZURE_BLOB,
        sharepoint_site_url: Optional[str] = None,
        sharepoint_auth_identity: Optional[str] = None,
        sharepoint_tenant_id: Optional[str] = None,
    ) -> SearchIndexerDataSourceConnection:
        """
        Create a data source connection (Azure Blob Storage or SharePoint Online).

        Args:
            data_source_name: Name for the data source
            container_name: For Blob Storage: container name; For SharePoint: library name
            data_source_type: Type of data source (AZURE_BLOB or SHAREPOINT)
            sharepoint_site_url: Required for SharePoint. Example: "https://tenant.sharepoint.com/sites/sitename"
            sharepoint_auth_identity: For SharePoint. Use "system" for managed identity (recommended).
                                     Leave None for connection_string auth (blob storage).
            sharepoint_tenant_id: Target Tenant ID for SharePoint (overrides pipeline default).

        Returns:
            Created SearchIndexerDataSourceConnection object
        """
        if data_source_type == DataSourceType.AZURE_BLOB:
            return self._create_blob_data_source(data_source_name, container_name)
        elif data_source_type == DataSourceType.SHAREPOINT:
            # Use passed tenant_id or fall back to pipeline configuration
            target_tenant_id = sharepoint_tenant_id or self.sharepoint_tenant_id

            return self._create_sharepoint_data_source(
                data_source_name,
                container_name,
                sharepoint_site_url,
                sharepoint_auth_identity,
                target_tenant_id,
            )
        else:
            raise ValueError(f"Unsupported data source type: {data_source_type}")

    def _create_blob_data_source(
        self, data_source_name: str, container_name: str
    ) -> SearchIndexerDataSourceConnection:
        """
        Create a data source connection to Azure Blob Storage.

        Args:
            data_source_name: Name for the data source
            container_name: Name of the blob storage container

        Returns:
            Created SearchIndexerDataSourceConnection object
        """
        container = SearchIndexerDataContainer(name=container_name)
        data_source_connection = SearchIndexerDataSourceConnection(
            name=data_source_name,
            type=DataSourceType.AZURE_BLOB.value,
            connection_string=self.azure_storage_connection,
            container=container,
        )

        result = self.indexer_client.create_or_update_data_source_connection(
            data_source_connection
        )
        print(f"✓ Blob Storage data source '{result.name}' created successfully")

        return result

    def _create_sharepoint_data_source(
        self,
        data_source_name: str,
        library_name: str,
        site_url: Optional[str] = None,
        auth_identity: Optional[str] = None,
        sharepoint_tenant_id: Optional[str] = None,
    ) -> SearchIndexerDataSourceConnection:
        """
        Create a data source connection to SharePoint Online.

        Uses the app registration credentials configured in Terraform (sharepoint_indexer_app_id/secret).

        SETUP REQUIRED:
        1. Create app registration in Azure AD with permissions:
           - Sites.Read.All (Application permission)
           - Files.Read.All (Application permission)
        2. Grant tenant admin consent
        3. Configure in terraform.tfvars.json:
           - sharepoint_indexer_app_id: "your-client-id"
           - sharepoint_indexer_app_secret: "your-client-secret"
        4. Run Terraform to assign roles

        Args:
            data_source_name: Name for the data source
            library_name: Name of the SharePoint document library
            site_url: SharePoint site URL. Example: "https://tenant.sharepoint.com/sites/sitename"
            auth_identity: Authentication method. Use "app" for app registration (or leave None, defaults to app)
            sharepoint_tenant_id: Target Tenant ID. Required for cross-tenant indexing.

        Returns:
            Created SearchIndexerDataSourceConnection object

        References:
            https://learn.microsoft.com/en-us/azure/search/search-howto-index-sharepoint-online
        """
        if not site_url:
            raise ValueError(
                "sharepoint_site_url is required for SharePoint data source. "
                "Example: https://tenant.sharepoint.com/sites/mysite"
            )

        if auth_identity == "system":
            # Use Managed Identity (System Assigned)
            # Requires ApplicationId but NO ApplicationSecret
            if not self.sharepoint_app_id:
                raise ValueError(
                    "SharePoint App ID is required even for Managed Identity. "
                    "Please configure sharepoint_indexer_app_id."
                )

            connection_string = (
                f"SharePointOnlineEndpoint={site_url};"
                f"ApplicationId={self.sharepoint_app_id};"
            )
            if sharepoint_tenant_id:
                connection_string += f"TenantId={sharepoint_tenant_id};"
        else:
            # Verify app registration credentials are available
            if not self.sharepoint_app_id or not self.sharepoint_app_secret:
                raise ValueError(
                    "SharePoint app registration credentials not found. "
                    "Please configure in terraform.tfvars.json:\n"
                    '  "sharepoint_indexer_app_id": "your-client-id",\n'
                    '  "sharepoint_indexer_app_secret": "your-client-secret"\n'
                    "Or set environment variables:\n"
                    "  SHAREPOINT_INDEXER_APP_ID\n"
                    "  SHAREPOINT_INDEXER_APP_SECRET"
                )

            # SharePoint connection string using app registration
            # Format: SharePointOnlineEndpoint={site_url};ApplicationId={app_id};ApplicationSecret={app_secret}
            connection_string = (
                f"SharePointOnlineEndpoint={site_url};"
                f"ApplicationId={self.sharepoint_app_id};"
                f"ApplicationSecret={self.sharepoint_app_secret};"
            )
            if sharepoint_tenant_id:
                connection_string += f"TenantId={sharepoint_tenant_id};"

        # For SharePoint, container.name must be one of these keywords:
        # - "defaultSiteLibrary": indexes the site's default document library
        # - "allSiteLibraries": indexes all document libraries in the site
        # - "useQuery": indexes content defined by the query parameter

        if library_name in ["defaultSiteLibrary", "allSiteLibraries"]:
            container = SearchIndexerDataContainer(name=library_name)
        else:
            # We use "useQuery" with includeLibrary to target a specific library by name
            # This allows library names with spaces (URL-encoded in the query)
            library_url = f"{site_url}/{library_name}".replace(" ", "%20")
            container = SearchIndexerDataContainer(
                name="useQuery", query=f"includeLibrary={library_url}"
            )

        data_source_connection = SearchIndexerDataSourceConnection(
            name=data_source_name,
            type=DataSourceType.SHAREPOINT.value,
            connection_string=connection_string,
            container=container,
        )

        result = self.indexer_client.create_or_update_data_source_connection(
            data_source_connection
        )

        auth_msg = (
            "Managed Identity"
            if auth_identity == "system"
            else f"app registration '{self.sharepoint_app_id}'"
        )
        print(
            f"✓ SharePoint data source '{result.name}' created successfully "
            f"using {auth_msg}."
        )

        return result

    def create_indexer(
        self,
        indexer_name: str,
        data_source_name: str,
        batch_size: int = 20,
        max_failed_items: int = -1,
        field_mappings: Optional[list[FieldMapping]] = None,
    ) -> SearchIndexer:
        """
        Create an indexer to process documents from the data source.

        Args:
            indexer_name: Name for the indexer
            data_source_name: Name of the data source to use
            batch_size: Number of documents to process in each batch
            max_failed_items: Maximum failed items before stopping (-1 for unlimited)
            field_mappings: Optional list of field mappings (Source -> Enriched Doc)

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
        )

        indexer = SearchIndexer(
            name=indexer_name,
            description="Indexer to index documents and generate embeddings",
            skillset_name=self.skillset_name,
            target_index_name=self.index.name,
            data_source_name=data_source_name,
            parameters=indexer_parameters,
            field_mappings=field_mappings,
        )

        result = self.indexer_client.create_or_update_indexer(indexer)
        print(
            f"✓ Indexer '{indexer_name}' created successfully and is running. "
            f"Give the indexer a few minutes to process documents."
        )

        return result

    def initialize_pipeline(
        self, data_source_type: DataSourceType = DataSourceType.AZURE_BLOB
    ) -> tuple[SearchIndex, SearchIndexerSkillset]:
        """
        Convenience method to initialize the index and skillset in one call.

        Args:
            data_source_type: Type of data source to configure skillset mappings

        Returns:
            Tuple of (index, skillset)
        """
        index = self.create_index()
        skillset = self.create_skillset(data_source_type=data_source_type)
        return index, skillset
