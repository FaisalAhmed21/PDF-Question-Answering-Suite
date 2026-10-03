"""Advanced NLP techniques for ingestion: Semantic Chunking, Doc2Query, Table Summarization."""

import asyncio
import logging
import math
from typing import List, Callable, Awaitable

from app.types import ElementType, RawElement, Chunk, ChunkType
from app.generation.llm_client import get_llm_client
from app.config import settings

logger = logging.getLogger(__name__)

# Try importing nltk for sentence tokenization
try:
    import nltk
    # We attempt to use it, if punkt is missing we will catch it below
except ImportError:
    pass


def _cosine_similarity(v1: List[float], v2: List[float]) -> float:
    """Compute cosine similarity between two vectors."""
    dot_product = sum(a * b for a, b in zip(v1, v2))
    norm_v1 = math.sqrt(sum(a * a for a in v1))
    norm_v2 = math.sqrt(sum(b * b for b in v2))
    if norm_v1 == 0 or norm_v2 == 0:
        return 0.0
    return dot_product / (norm_v1 * norm_v2)


async def semantic_chunking(
    text: str, 
    embedder_func: Callable[[List[str]], Awaitable[List[List[float]]]]
) -> List[str]:
    """
    Split text into sentences, embed them, and group into chunks based on semantic similarity.
    Creates a new chunk when the cosine similarity between adjacent sentences drops significantly.
    """
    if not text.strip():
        return []

    # 1. Split into sentences
    try:
        from nltk.tokenize import sent_tokenize
        try:
            sentences = sent_tokenize(text)
        except LookupError:
            import nltk
            nltk.download('punkt')
            nltk.download('punkt_tab')
            sentences = sent_tokenize(text)
    except ImportError:
        logger.warning("nltk not installed. Falling back to naive sentence splitting.")
        sentences = [s.strip() + "." for s in text.replace("!", ".").replace("?", ".").split(".") if s.strip()]

    if len(sentences) <= 1:
        return sentences

    # 2. Embed sentences
    try:
        embeddings = await embedder_func(sentences)
    except Exception as e:
        logger.error(f"Semantic chunking embed failed, returning full text: {e}")
        return [text]

    # 3. Calculate similarities between consecutive sentences
    similarities = []
    for i in range(len(embeddings) - 1):
        sim = _cosine_similarity(embeddings[i], embeddings[i+1])
        similarities.append(sim)

    # If no similarities (only 1 or 0 sentences), return sentences
    if not similarities:
        return [" ".join(sentences)]

    # 4. Determine splitting threshold (e.g., lower 15th percentile / 85th percentile of drops)
    # We want to split where similarity is anomalously low.
    sorted_sims = sorted(similarities)
    # The 15th percentile of similarities (meaning the bottom 15% most dissimilar adjacent sentences)
    split_index = max(0, int(len(sorted_sims) * 0.15))
    threshold = sorted_sims[split_index]

    # 5. Group sentences into chunks
    chunks = []
    current_chunk = [sentences[0]]

    for i in range(len(similarities)):
        if similarities[i] < threshold:
            # Topic shift detected, flush current chunk
            chunks.append(" ".join(current_chunk))
            current_chunk = [sentences[i+1]]
        else:
            # Keep in the same chunk
            current_chunk.append(sentences[i+1])

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks


async def generate_doc2query(chunk_text: str) -> str:
    """
    Generate 3 specific questions that the text chunk can answer.
    Returns a string with the questions to be appended to contextualized_content.
    """
    prompt = (
        "Given this text chunk, generate 3 specific questions that this text can directly answer. "
        "Output ONLY the questions separated by newlines, with no markdown formatting or intro text.\n\n"
        f"Text:\n{chunk_text}"
    )

    try:
        client = get_llm_client()
        result = await client.generate(
            prompt=prompt,
        )
        return result.strip()
    except Exception as e:
        logger.warning(f"Doc2Query generation failed: {e}")
        return ""


async def summarize_table(table_md: str) -> str:
    """
    Summarize a raw markdown table into a 2-paragraph plain-text description.
    """
    prompt = (
        "You are a data analyst. Read this markdown table and write a 2-paragraph plain-text "
        "summary of the key trends, data points, and relationships it contains. "
        "Do not include the markdown table itself in the output, just the summary.\n\n"
        f"Table:\n{table_md}"
    )

    try:
        client = get_llm_client()
        result = await client.generate(
            prompt=prompt,
        )
        return result.strip()
    except Exception as e:
        logger.warning(f"Table summarization failed: {e}")
        # Fall back to returning the raw markdown if the LLM fails
        return table_md
