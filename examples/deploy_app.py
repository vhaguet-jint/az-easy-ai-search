import os
import sys
from datetime import datetime

import streamlit as st
from dotenv import load_dotenv

# Add repo root to path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from jint_search_pipelines import (
    DataSourceType,
    HybridSearchPipeline,
    HybridSemanticOCRSearchPipeline,
    HybridSemanticSearchPipeline,
    OCRImageSearchPipeline,
    VectorSearchPipeline,
)

load_dotenv()

# ============================================================================
# PIPELINE CONFIGURATIONS
# ============================================================================

PIPELINE_TYPES = {
    "Vector Search (PDF)": {
        "class": VectorSearchPipeline,
        "description": "Basic vector search for PDF documents",
        "supports_ocr": False,
        "has_semantic": False,
        "has_hybrid": False,
    },
    "Hybrid Search (PDF)": {
        "class": HybridSearchPipeline,
        "description": "Vector + keyword search for PDF documents",
        "supports_ocr": False,
        "has_semantic": False,
        "has_hybrid": True,
    },
    "Hybrid + Semantic (PDF)": {
        "class": HybridSemanticSearchPipeline,
        "description": "Vector + keyword + semantic reranking for PDF documents",
        "supports_ocr": False,
        "has_semantic": True,
        "has_hybrid": True,
    },
    "OCR Image Search": {
        "class": OCRImageSearchPipeline,
        "description": "OCR + vector search for PNG/JPEG images",
        "supports_ocr": True,
        "has_semantic": False,
        "has_hybrid": False,
    },
    "Hybrid + Semantic + OCR (Images)": {
        "class": HybridSemanticOCRSearchPipeline,
        "description": "OCR + vector + keyword + semantic reranking for images",
        "supports_ocr": True,
        "has_semantic": True,
        "has_hybrid": True,
    },
}

DATA_SOURCE_TYPES = {
    "Azure Blob Storage": DataSourceType.AZURE_BLOB,
    "SharePoint Online": DataSourceType.SHAREPOINT,
}

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================

st.set_page_config(
    page_title="Azure Search Pipeline Deployer",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .main {
        padding: 2rem;
    }
    .stTabs [data-baseweb="tab-list"] button [data-testid="stMarkdownContainer"] p {
        font-size: 1rem;
        font-weight: 600;
    }
    </style>
""",
    unsafe_allow_html=True,
)

# ============================================================================
# MAIN APP
# ============================================================================

st.title("🔍 Azure Search Pipeline Deployer")
st.markdown("""
Deploy and configure Azure AI Search indexes with different search capabilities.
Choose a pipeline type, data source, and provide your configuration.
""")

# Sidebar for general settings
with st.sidebar:
    st.header("⚙️ Configuration")

    st.markdown("### Azure Credentials")
    st.info("""
    ℹ️ **Note:** This app uses Azure DefaultAzureCredential.
    Make sure you have Azure CLI authenticated or use service principal env vars.
    """)

    # Check environment variables
    env_vars = {
        "AZURE_SEARCH_SERVICE": os.getenv("AZURE_SEARCH_SERVICE"),
        "AZURE_OPENAI_ACCOUNT": os.getenv("AZURE_OPENAI_ACCOUNT"),
        "AZURE_STORAGE_CONNECTION": os.getenv("AZURE_STORAGE_CONNECTION"),
    }

    all_set = all(v for v in env_vars.values())

    if all_set:
        st.success("✅ All required environment variables are set")
    else:
        st.warning("⚠️ Missing environment variables:")
        for key, value in env_vars.items():
            status = "✅" if value else "❌"
            st.text(f"{status} {key}")

# Pipeline type selection
st.header("Pipeline Configuration")
selected_pipeline = st.selectbox(
    "🔧 Select Pipeline Type",
    options=list(PIPELINE_TYPES.keys()),
    help="Choose the search pipeline that best fits your use case",
    index=0,
)

pipeline_config = PIPELINE_TYPES[selected_pipeline]
st.markdown(f"**Description:** {pipeline_config['description']}")

# ============================================================================
# DATA SOURCE SECTION
# ============================================================================

st.markdown("---")
st.header("📦 Data Source")

selected_source = st.selectbox(
    "Select Data Source Type",
    options=list(DATA_SOURCE_TYPES.keys()),
    index=0,
)
source_type = DATA_SOURCE_TYPES[selected_source]

if source_type == DataSourceType.AZURE_BLOB:
    blob_container = st.text_input(
        "Blob Container Name",
        value="documents",
        help="Name of the container in Azure Blob Storage",
    )

    source_params = {
        "container_name": blob_container,
        "data_source_type": DataSourceType.AZURE_BLOB,
    }

else:  # SharePoint
    col1, col2 = st.columns(2)

    with col1:
        sharepoint_site_url = st.text_input(
            "SharePoint Site URL",
            value="https://example.sharepoint.com/sites/MyLibrary",
            help="Full URL of the SharePoint site",
        )

    with col2:
        sharepoint_library = st.text_input(
            "Document Library Name",
            value="Shared Documents",
            help="Name of the document library in SharePoint",
        )

    sharepoint_tenant_id = st.text_input(
        "SharePoint Tenant ID",
        value="",
        type="default",
        help="The Azure AD Tenant ID where SharePoint data resides",
    )

    source_params = {
        "container_name": sharepoint_library,
        "data_source_type": DataSourceType.SHAREPOINT,
        "sharepoint_site_url": sharepoint_site_url,
        "sharepoint_auth_identity": None,
        "sharepoint_tenant_id": sharepoint_tenant_id,
    }

# ============================================================================
# CONFIGURATION SECTION
# ============================================================================

st.markdown("---")
st.header("⚙️ Configuration")

col1, col2 = st.columns([2, 1])

with col1:
    project_name = st.text_input(
        "Project Name",
        value="my-project",
        help="Used to generate simplified names for index, skillset, indexer, data source",
    )

with col2:
    timestamp = st.checkbox(
        "Add Timestamp",
        value=False,
        help="Append timestamp to names for uniqueness",
    )


def generate_names(project: str, add_timestamp: bool = False):
    suffix = f"-{datetime.now().strftime('%Y%m%d-%H%M%S')}" if add_timestamp else ""
    return (
        f"{project}-idx{suffix}",
        f"{project}-ss{suffix}",
        f"{project}-idxr{suffix}",
        f"{project}-ds{suffix}",
    )


index_name, skillset_name, indexer_name, data_source_name = generate_names(
    project_name, timestamp
)

st.markdown("---")
st.subheader("Embedding & Chunking Settings")

col1, col2 = st.columns(2)

with col1:
    embedding_deployment = st.text_input(
        "Embedding Deployment Name",
        value="text-embedding-3-large",
        help="Name of the Azure OpenAI embedding deployment",
    )

with col2:
    embedding_dimensions = st.number_input(
        "Embedding Dimensions",
        value=3072,
        min_value=384,
        step=1,
        help="Vector dimensionality (e.g., 1536 for text-embedding-3-small, 3072 for text-embedding-3-large)",
    )
    st.warning(
        "⚠️ Must match the dimensions of your embedding model. "
        "Use 1536 for text-embedding-3-small, 3072 for text-embedding-3-large",
        icon="⚠️",
    )

col1, col2 = st.columns(2)

with col1:
    chunk_size = st.number_input(
        "Chunk Size (characters)",
        value=2000,
        min_value=500,
        step=500,
        help="Maximum length of each text chunk",
    )

with col2:
    chunk_overlap = st.number_input(
        "Chunk Overlap (characters)",
        value=500,
        min_value=0,
        step=100,
        help="Overlap between consecutive chunks for context preservation",
    )

# ============================================================================
# DEPLOYMENT BUTTON & EXECUTION
# ============================================================================

st.markdown("---")

col1, col2, col3 = st.columns([1, 1, 2])

with col1:
    deploy_button = st.button(
        "🚀 Deploy Pipeline",
        type="primary",
        use_container_width=True,
        help="Deploy the configured search pipeline to Azure",
    )

with col2:
    preview_button = st.button(
        "👁️ Preview Configuration",
        use_container_width=True,
        help="Show the configuration that will be used",
    )

if preview_button:
    st.markdown("### Configuration Preview")
    preview_config = {
        "Pipeline Type": selected_pipeline,
        "Data Source": selected_source,
        "Project Name": project_name,
        "Index Name": index_name,
        "Skillset Name": skillset_name,
        "Indexer Name": indexer_name,
        "Data Source Name": data_source_name,
        "Embedding Deployment": embedding_deployment,
        "Embedding Dimensions": embedding_dimensions,
        "Chunk Size": chunk_size,
        "Chunk Overlap": chunk_overlap,
        **source_params,
    }
    st.json({k: str(v) for k, v in preview_config.items()})

if deploy_button:
    if not all(
        [
            env_vars.get("AZURE_SEARCH_SERVICE"),
            env_vars.get("AZURE_OPENAI_ACCOUNT"),
            env_vars.get("AZURE_STORAGE_CONNECTION"),
        ]
    ):
        st.error("❌ Missing required environment variables. Please check sidebar.")
    else:
        try:
            with st.spinner("🔄 Deploying pipeline..."):
                pipeline_class = pipeline_config["class"]

                pipeline_kwargs = {
                    "index_name": index_name,
                    "skillset_name": skillset_name,
                    "embedding_deployment": embedding_deployment,
                    "embedding_dimensions": int(embedding_dimensions),
                    "chunk_size": int(chunk_size),
                    "chunk_overlap": int(chunk_overlap),
                }

                if source_type == DataSourceType.SHAREPOINT and sharepoint_tenant_id:
                    pipeline_kwargs["sharepoint_tenant_id"] = sharepoint_tenant_id

                pipeline = pipeline_class(**pipeline_kwargs)

                st.info("📝 Initializing pipeline...")
                index, skillset = pipeline.initialize_pipeline(
                    data_source_type=source_type
                )
                st.success("✅ Pipeline initialized")

                st.info("📊 Creating data source...")
                data_src = pipeline.create_data_source(
                    data_source_name=data_source_name,
                    **source_params,
                )
                st.success(f"✅ Data source created: {data_src.name}")

                st.info("⚙️ Creating indexer...")
                indexer = pipeline.create_indexer(
                    indexer_name=indexer_name,
                    data_source_name=data_source_name,
                )
                st.success(f"✅ Indexer created: {indexer.name}")

                st.success("🎉 Pipeline deployed successfully!")

                st.markdown("### 📊 Deployment Summary")
                summary = {
                    "Index": index_name,
                    "Skillset": skillset_name,
                    "Indexer": indexer_name,
                    "Data Source": data_source_name,
                    "Pipeline Type": selected_pipeline,
                    "Data Source Type": selected_source,
                    "Timestamp": datetime.now().isoformat(),
                }
                st.json(summary)
                st.balloons()

        except Exception as e:
            st.error(f"❌ Deployment failed: {str(e)}")
            st.exception(e)
