from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from helpers.custom_lcdoctransformers import EmbeddingsClusteringFilter
from helpers.constants import LARGE_EMBD

EMBEDDER = OpenAIEmbeddings(model=LARGE_EMBD)

async def a_cluster_strings(strings: list[str]):
    num_clusters = 1
    cluster_size = len(strings)
    clusterer = EmbeddingsClusteringFilter(
        embeddings=EMBEDDER,
        num_clusters=num_clusters,
        num_closest=cluster_size
    )
    transformed = await clusterer.atransform_documents([Document(page_content=string) for string in strings])
    transformed = [statedoc.page_content for statedoc in transformed]
    return transformed