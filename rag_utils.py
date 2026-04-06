import os

# Disable Chroma/Chromadb anonymized telemetry to avoid noisy console warnings and
# prevent client telemetry codepaths from running inside some Docker environments.
os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")
os.environ.setdefault("CHROMA_ANONYMIZED_TELEMETRY", "False")
os.environ.setdefault("POSTHOG_DISABLED", "1")
import shutil
import json
import time
from dataclasses import dataclass
from typing import List, Optional, Dict, Any

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_core.documents import Document

from chromadb.config import Settings as ChromaSettings

try:
    from langchain_ollama import ChatOllama, OllamaEmbeddings
except Exception:
    ChatOllama = None
    OllamaEmbeddings = None


DEFAULT_PERSIST_DIRECTORY = os.getenv("RAG_PERSIST_DIRECTORY", "./chroma_url_db")
DEFAULT_PROVIDER = os.getenv("RAG_PROVIDER", "ollama")
DEFAULT_OLLAMA_BASE_URL = os.getenv(
    "OLLAMA_BASE_URL",
    os.getenv("RAG_OLLAMA_BASE_URL", "http://localhost:11434"),
)
DEFAULT_OLLAMA_CHAT_MODEL = os.getenv("RAG_OLLAMA_CHAT_MODEL", "llama3.1")
DEFAULT_CHAT_MODEL = DEFAULT_OLLAMA_CHAT_MODEL
DEFAULT_OLLAMA_EMBEDDING_MODEL = os.getenv(
    "RAG_OLLAMA_EMBEDDING_MODEL", "nomic-embed-text"
)
DEFAULT_EMBEDDING_MODEL = DEFAULT_OLLAMA_EMBEDDING_MODEL
DEFAULT_CHUNK_SIZE = int(os.getenv("RAG_CHUNK_SIZE", "1000"))
DEFAULT_CHUNK_OVERLAP = int(os.getenv("RAG_CHUNK_OVERLAP", "200"))
DEFAULT_K = int(os.getenv("RAG_TOP_K", "4"))
DEFAULT_COLLECTION_NAME = os.getenv("RAG_COLLECTION_NAME", "url_docs")
DEFAULT_ARTIFACTS_DIR = os.getenv("RAG_ARTIFACTS_DIR", "./rag_artifacts")


def env_flag(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "t", "yes", "y", "on"}


@dataclass(frozen=True)
class RAGSettings:
    persist_directory: str = DEFAULT_PERSIST_DIRECTORY
    provider: str = DEFAULT_PROVIDER
    embedding_model: str = DEFAULT_EMBEDDING_MODEL
    chat_model: str = DEFAULT_CHAT_MODEL
    ollama_base_url: str = DEFAULT_OLLAMA_BASE_URL
    ollama_chat_model: str = DEFAULT_OLLAMA_CHAT_MODEL
    ollama_embedding_model: str = DEFAULT_OLLAMA_EMBEDDING_MODEL
    chunk_size: int = DEFAULT_CHUNK_SIZE
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP
    top_k: int = DEFAULT_K
    collection_name: str = DEFAULT_COLLECTION_NAME
    artifacts_dir: str = DEFAULT_ARTIFACTS_DIR
    reset_db: bool = False


def settings_from_dict(d: Dict[str, Any], base: Optional[RAGSettings] = None) -> RAGSettings:
    b = base or RAGSettings()
    return RAGSettings(
        persist_directory=str(d.get("persist_directory", b.persist_directory)),
        provider=str(d.get("provider", b.provider)),
        embedding_model=str(d.get("embedding_model", b.embedding_model)),
        chat_model=str(d.get("chat_model", b.chat_model)),
        ollama_base_url=str(d.get("ollama_base_url", b.ollama_base_url)),
        ollama_chat_model=str(d.get("ollama_chat_model", b.ollama_chat_model)),
        ollama_embedding_model=str(
            d.get("ollama_embedding_model", b.ollama_embedding_model)
        ),
        chunk_size=int(d.get("chunk_size", b.chunk_size)),
        chunk_overlap=int(d.get("chunk_overlap", b.chunk_overlap)),
        top_k=int(d.get("top_k", b.top_k)),
        collection_name=str(d.get("collection_name", b.collection_name)),
        artifacts_dir=str(d.get("artifacts_dir", b.artifacts_dir)),
        reset_db=bool(d.get("reset_db", b.reset_db)),
    )


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def docs_to_jsonable(docs: List[Document]) -> List[dict]:
    items: List[dict] = []
    for d in docs:
        items.append({"page_content": d.page_content, "metadata": d.metadata or {}})
    return items


def docs_from_jsonable(items: List[dict]) -> List[Document]:
    docs: List[Document] = []
    for it in items:
        docs.append(
            Document(
                page_content=it.get("page_content", ""),
                metadata=it.get("metadata", {}) or {},
            )
        )
    return docs


def artifacts_paths(settings: RAGSettings, name: str = "url") -> dict:
    base = os.path.join(settings.artifacts_dir, name)
    return {
        "base": base,
        "docs": os.path.join(base, "documents.json"),
        "chunks": os.path.join(base, "chunks.json"),
    }


def save_documents(docs: List[Document], path: str) -> None:
    ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as f:
        json.dump(docs_to_jsonable(docs), f, ensure_ascii=False, indent=2)


def load_documents(path: str) -> List[Document]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return docs_from_jsonable(data)


def validate_env(settings: RAGSettings) -> None:
    if settings.provider.lower() == "ollama" and OllamaEmbeddings is None:
        raise ImportError(
            "No se encontró langchain_ollama. Instálalo con: pip install langchain-ollama"
        )


def get_embeddings(settings: RAGSettings):
    provider = settings.provider.lower()
    if provider == "ollama":
        if OllamaEmbeddings is None:
            raise ImportError(
                "No se encontró langchain_ollama. Instálalo con: pip install langchain-ollama"
            )
        return OllamaEmbeddings(
            model=settings.ollama_embedding_model,
            base_url=settings.ollama_base_url,
        )

    raise ValueError(
        f"Proveedor no soportado: {provider}. Solo se soporta 'ollama'."
    )


def get_llm(settings: RAGSettings):
    provider = settings.provider.lower()
    if provider == "ollama":
        if ChatOllama is None:
            raise ImportError(
                "No se encontró langchain_ollama. Instálalo con: pip install langchain-ollama"
            )
        return ChatOllama(
            model=settings.ollama_chat_model,
            base_url=settings.ollama_base_url,
        )

    raise ValueError(
        f"Proveedor no soportado: {provider}. Solo se soporta 'ollama'."
    )


def split_documents(documents: List[Document], settings: RAGSettings) -> List[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    return splitter.split_documents(documents)


def build_vectorstore(chunks: List[Document], settings: RAGSettings):
    if settings.reset_db and os.path.exists(settings.persist_directory):
        shutil.rmtree(settings.persist_directory)

    chroma_settings = ChromaSettings(anonymized_telemetry=False)

    t0 = time.perf_counter()
    print(
        f"    [EMBEDDINGS] provider={settings.provider} "
        f"model={(settings.ollama_embedding_model if settings.provider.lower() == 'ollama' else settings.embedding_model)}"
    )
    embeddings = get_embeddings(settings)
    print(f"    [EMBEDDINGS] init ok ({time.perf_counter() - t0:.2f}s)")

    if os.path.exists(settings.persist_directory):
        vectorstore = Chroma(
            persist_directory=settings.persist_directory,
            embedding_function=embeddings,
            collection_name=settings.collection_name,
            client_settings=chroma_settings,
        )
        t1 = time.perf_counter()
        print(f"    [INDEX] add_documents chunks={len(chunks)}")
        vectorstore.add_documents(chunks)
        print(f"    [INDEX] add_documents ok ({time.perf_counter() - t1:.2f}s)")
        return vectorstore

    t1 = time.perf_counter()
    print(f"    [INDEX] from_documents chunks={len(chunks)}")
    vs = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=settings.persist_directory,
        collection_name=settings.collection_name,
        client_settings=chroma_settings,
    )
    print(f"    [INDEX] from_documents ok ({time.perf_counter() - t1:.2f}s)")
    return vs


def index_chunks(chunks: List[Document], settings: RAGSettings) -> None:
    build_vectorstore(chunks, settings)


def load_vectorstore(settings: RAGSettings):
    if not os.path.exists(settings.persist_directory):
        raise FileNotFoundError(
            f"No existe la base vectorial en: {settings.persist_directory}"
        )

    embeddings = get_embeddings(settings)
    chroma_settings = ChromaSettings(anonymized_telemetry=False)
    return Chroma(
        persist_directory=settings.persist_directory,
        embedding_function=embeddings,
        collection_name=settings.collection_name,
        client_settings=chroma_settings,
    )
