import os
import argparse

import rag_utils
from utils.common import load_config_file


DEFAULT_CONFIG_PATH = "./config/rag_config.json"


def env_override_present() -> bool:
    return os.getenv("OLLAMA_BASE_URL") is not None or os.getenv("RAG_OLLAMA_BASE_URL") is not None


def build_settings(args: argparse.Namespace) -> rag_utils.RAGSettings:
    fallback = "./rag_config.json" if args.config == DEFAULT_CONFIG_PATH else None
    config = load_config_file(args.config, fallback_path=fallback)
    base = rag_utils.settings_from_dict(config.get("rag", {}) if isinstance(config, dict) else {})

    if env_override_present() and args.ollama_base_url is None:
        base = rag_utils.settings_from_dict({"ollama_base_url": rag_utils.DEFAULT_OLLAMA_BASE_URL}, base=base)

    return rag_utils.settings_from_dict(
        {
            "persist_directory": args.persist_directory or base.persist_directory,
            "artifacts_dir": args.artifacts_dir or base.artifacts_dir,
            "provider": args.provider or base.provider,
            "ollama_base_url": args.ollama_base_url or base.ollama_base_url,
            "ollama_chat_model": args.ollama_chat_model or base.ollama_chat_model,
            "ollama_embedding_model": args.ollama_embedding_model or base.ollama_embedding_model,
            "collection_name": args.collection_name or base.collection_name,
            "embedding_model": args.embedding_model or base.embedding_model,
            "chat_model": args.chat_model or base.chat_model,
            "chunk_size": args.chunk_size if args.chunk_size is not None else base.chunk_size,
            "chunk_overlap": args.chunk_overlap if args.chunk_overlap is not None else base.chunk_overlap,
            "top_k": args.k if args.k is not None else base.top_k,
            "reset_db": bool(args.reset_db) or rag_utils.env_flag("RAG_RESET_DB", False) or base.reset_db,
        },
        base=base,
    )


def main():
    parser = argparse.ArgumentParser(
        description="Runner: ejecuta secuencialmente fetch -> chunk -> index y opcionalmente ask/chat."
    )

    parser.add_argument(
        "--config",
        default=DEFAULT_CONFIG_PATH,
        help="Archivo de configuración JSON (por defecto: ./config/rag_config.json)",
    )
    parser.add_argument("--url", required=True, help="URL pública a indexar")
    parser.add_argument("--question", help="Si se provee, ejecuta ask al final")
    parser.add_argument("--chat", action="store_true", help="Si se activa, entra en modo chat al final")

    parser.add_argument(
        "--provider",
        choices=["ollama"],
        default=None,
        help="Sobrescribe provider del config (solo ollama)",
    )
    parser.add_argument("--persist-directory", default=None)
    parser.add_argument("--artifacts-dir", default=None)
    parser.add_argument("--collection-name", default=None)

    parser.add_argument("--ollama-base-url", default=None)
    parser.add_argument("--ollama-chat-model", default=None)
    parser.add_argument("--ollama-embedding-model", default=None)

    parser.add_argument("--embedding-model", default=None)
    parser.add_argument("--chat-model", default=None)

    parser.add_argument("--chunk-size", type=int, default=None)
    parser.add_argument("--chunk-overlap", type=int, default=None)
    parser.add_argument("--k", type=int, default=None)

    parser.add_argument("--reset-db", action="store_true")

    args = parser.parse_args()
    settings = build_settings(args)
    rag_utils.validate_env(settings)

    paths = rag_utils.artifacts_paths(settings, name="url")

    print(f"\n[1/3] FETCH: {args.url}")
    docs = rag_utils.fetch_and_clean(args.url)
    rag_utils.save_documents(docs, paths["docs"])
    print(f"    OK -> {paths['docs']} (docs={len(docs)})")

    print("\n[2/3] CHUNK")
    chunks = rag_utils.chunk_documents(docs, settings)
    rag_utils.save_documents(chunks, paths["chunks"])
    print(f"    OK -> {paths['chunks']} (chunks={len(chunks)})")

    print("\n[3/3] INDEX")
    rag_utils.index_chunks(chunks, settings)
    print(f"    OK -> Chroma en {settings.persist_directory}")

    if args.question:
        rag_utils.ask_question(args.question, settings)

    if args.chat:
        rag_utils.interactive_mode(settings)


if __name__ == "__main__":
    main()
