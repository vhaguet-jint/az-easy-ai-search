from dotenv import load_dotenv

from .AzureSearchPipelineBase import AzureSearchPipelineBase

load_dotenv()


class VectorSearchPipeline(AzureSearchPipelineBase):
    """
    Pipeline for basic vector search only.

    This is the simplest search pipeline that uses only vector embeddings
    for semantic similarity search.
    """

    pass
