# az-easy-ai-search

Python helpers for deploying Azure AI Search indexes with different search strategies.
Wraps the Azure SDK to reduce boilerplate — pick a pipeline class, call three methods, done.

## Pipelines

| Class | Use case | Vector | Keyword | Semantic | OCR |
|---|---|---|---|---|---|
| `VectorSearchPipeline` | PDF | ✓ | | | |
| `HybridSearchPipeline` | PDF | ✓ | ✓ | | |
| `HybridSemanticSearchPipeline` | PDF | ✓ | ✓ | ✓ | |
| `OCRImageSearchPipeline` | Images (PNG/JPEG) | ✓ | ✓ | | ✓ |
| `HybridSemanticOCRSearchPipeline` | Images (PNG/JPEG) | ✓ | ✓ | ✓ | ✓ |

Data sources: **Azure Blob Storage** and **SharePoint Online**.

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env  # fill in your values
```

Authenticate with Azure CLI:
```bash
az login
```

## Usage

```python
from jint_search_pipelines import HybridSemanticSearchPipeline, DataSourceType

pipeline = HybridSemanticSearchPipeline(
    index_name="my-idx",
    skillset_name="my-ss",
)

pipeline.initialize_pipeline(data_source_type=DataSourceType.AZURE_BLOB)

pipeline.create_data_source(
    data_source_name="my-ds",
    container_name="documents",
)

pipeline.create_indexer(
    indexer_name="my-idxr",
    data_source_name="my-ds",
)
```

### SharePoint

```python
pipeline.create_data_source(
    data_source_name="my-sp-ds",
    container_name="Shared Documents",
    data_source_type=DataSourceType.SHAREPOINT,
    sharepoint_site_url="https://tenant.sharepoint.com/sites/mysite",
    sharepoint_tenant_id="<tenant-id>",
)
```

## Environment variables

| Variable | Required | Description |
|---|---|---|
| `AZURE_SEARCH_SERVICE` | ✓ | Search service endpoint |
| `AZURE_OPENAI_ACCOUNT` | ✓ | Azure OpenAI endpoint |
| `AZURE_STORAGE_CONNECTION` | ✓ | Storage connection string |
| `AZURE_AI_SERVICES_KEY` | OCR only | AI Services key for OCR pipelines |
| `SHAREPOINT_INDEXER_APP_ID` | SharePoint | App registration client ID |
| `SHAREPOINT_INDEXER_APP_SECRET` | SharePoint | App registration secret |
| `SHAREPOINT_TENANT_ID` | SharePoint | Tenant ID |

Credentials are resolved in order: constructor args → env vars → `terraform/terraform.tfvars.json`.

## Examples

- `examples/deploy_app.py` — Streamlit UI for deploying pipelines
- `examples/rag_chat.py` — minimal RAG chat loop using a deployed index

```bash
streamlit run examples/deploy_app.py
```
