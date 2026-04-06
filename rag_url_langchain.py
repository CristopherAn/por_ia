import os
import argparse
import time

import rag_utils
from utils.common import load_config_file


DEFAULT_CONFIG_PATH = "./config/rag_config.json"


def fmt_s(seconds: float) -> str:
    return f"{seconds:.2f}s"


def log_step(label: str) -> float:
    print(f"\n[STEP] {label}")
    return time.perf_counter()


def log_done(t0: float, label: str = "done") -> float:
    dt = time.perf_counter() - t0
    print(f"    {label} ({fmt_s(dt)})")
    return dt


def resolve_url(args, config: dict) -> str:
    # Resuelve la URL objetivo con esta precedencia:
    # 1) --url (CLI)
    # 2) config: rag.url
    # 3) fallback: primer match.job_urls[0] (útil para reutilizar una URL de job del pipeline de matching)
    url = getattr(args, "url", None)
    if url:
        return url
    rag = config.get("rag", {}) if isinstance(config, dict) else {}
    cfg_url = rag.get("url")
    if cfg_url:
        return cfg_url
    match = config.get("match", {}) if isinstance(config, dict) else {}
    job_urls = match.get("job_urls")
    if isinstance(job_urls, list) and len(job_urls) > 0 and job_urls[0]:
        return job_urls[0]
    raise ValueError(
        "No se encontró URL. Pasá --url o definí rag.url en config/rag_config.json"
    )


def main():

    parser = argparse.ArgumentParser(
        description="RAG con LangChain usando una URL pública (OpenAI u Ollama)."
    )

    parser.add_argument(
        "--config",
        default=DEFAULT_CONFIG_PATH,
        help="Archivo de configuración JSON (por defecto: ./config/rag_config.json)",
    )

    parser.add_argument(
        "--persist-directory",
        default=rag_utils.DEFAULT_PERSIST_DIRECTORY,
        help="Directorio de persistencia de Chroma (env: RAG_PERSIST_DIRECTORY)",
    )
    parser.add_argument(
        "--artifacts-dir",
        default=rag_utils.DEFAULT_ARTIFACTS_DIR,
        help="Directorio para artefactos intermedios (env: RAG_ARTIFACTS_DIR)",
    )
    parser.add_argument(
        "--provider",
        default=rag_utils.DEFAULT_PROVIDER,
        choices=["openai", "ollama"],
        help="Proveedor LLM/embeddings: openai u ollama (env: RAG_PROVIDER)",
    )
    parser.add_argument(
        "--ollama-base-url",
        default=rag_utils.DEFAULT_OLLAMA_BASE_URL,
        help="Base URL de Ollama (env: OLLAMA_BASE_URL o RAG_OLLAMA_BASE_URL)",
    )
    parser.add_argument(
        "--ollama-chat-model",
        default=rag_utils.DEFAULT_OLLAMA_CHAT_MODEL,
        help="Modelo de chat en Ollama (env: RAG_OLLAMA_CHAT_MODEL)",
    )
    parser.add_argument(
        "--ollama-embedding-model",
        default=rag_utils.DEFAULT_OLLAMA_EMBEDDING_MODEL,
        help="Modelo de embeddings en Ollama (env: RAG_OLLAMA_EMBEDDING_MODEL)",
    )
    parser.add_argument(
        "--collection-name",
        default=rag_utils.DEFAULT_COLLECTION_NAME,
        help="Nombre de colección Chroma (env: RAG_COLLECTION_NAME)",
    )
    parser.add_argument(
        "--embedding-model",
        default=rag_utils.DEFAULT_EMBEDDING_MODEL,
        help="Modelo de embeddings OpenAI (env: RAG_EMBEDDING_MODEL)",
    )
    parser.add_argument(
        "--chat-model",
        default=rag_utils.DEFAULT_CHAT_MODEL,
        help="Modelo de chat OpenAI (env: RAG_CHAT_MODEL)",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=rag_utils.DEFAULT_CHUNK_SIZE,
        help="Tamaño de chunk (env: RAG_CHUNK_SIZE)",
    )
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=rag_utils.DEFAULT_CHUNK_OVERLAP,
        help="Overlap de chunk (env: RAG_CHUNK_OVERLAP)",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=rag_utils.DEFAULT_K,
        help="Top-k del retriever (env: RAG_TOP_K)",
    )
    parser.add_argument(
        "--reset-db",
        action="store_true",
        help="Elimina la BD antes de indexar (env: RAG_RESET_DB=1)",
    )
    parser.add_argument(
        "--auto-clean",
        action="store_true",
        default=True,
        help="Si ocurre error por artifacts inválidos/login wall, limpia artifacts automáticamente",
    )
    parser.add_argument(
        "--clean-db-on-error",
        action="store_true",
        help="Si ocurre error por artifacts inválidos/login wall, también borra persist_directory",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    # fetch: descarga y limpia la URL, guardando documents.json en artifacts_dir.
    # chunk: lee documents.json y genera chunks.json (chunking/splitting).
    # index: genera embeddings de chunks.json y persiste Chroma en persist_directory.
    # ask/chat: consultas sobre la BD vectorial persistida (requiere que index ya esté hecho).

    fetch_parser = subparsers.add_parser(
        "fetch",
        help="Descarga+limpia una URL y guarda documents.json",
    )
    fetch_parser.add_argument(
        "--url",
        required=False,
        default=None,
        help="URL pública (si no se indica, se toma de config: rag.url)",
    )

    subparsers.add_parser(
        "chunk",
        help="Lee documents.json y genera chunks.json",
    )

    index_parser = subparsers.add_parser(
        "index",
        help="Indexa una URL (si hay chunks.json, lo reutiliza)",
    )
    index_parser.add_argument(
        "--url",
        required=False,
        default=None,
        help="URL pública (si no se indica, se toma de config: rag.url)",
    )

    ask_parser = subparsers.add_parser("ask", help="Hace una pregunta")
    ask_parser.add_argument("--question", required=True, help="Pregunta")

    subparsers.add_parser("chat", help="Modo interactivo")

    args = parser.parse_args()
    fallback = "./rag_config.json" if args.config == DEFAULT_CONFIG_PATH else None
    t0 = log_step(f"load config -> {args.config}")
    config = load_config_file(args.config, fallback_path=fallback)
    log_done(t0)

    settings = rag_utils.settings_from_dict(
        {
            **(config.get("rag", {}) if isinstance(config, dict) else {}),
            "persist_directory": args.persist_directory,
            "artifacts_dir": args.artifacts_dir,
            "provider": args.provider,
            "ollama_base_url": args.ollama_base_url,
            "ollama_chat_model": args.ollama_chat_model,
            "ollama_embedding_model": args.ollama_embedding_model,
            "collection_name": args.collection_name,
            "embedding_model": args.embedding_model,
            "chat_model": args.chat_model,
            "chunk_size": args.chunk_size,
            "chunk_overlap": args.chunk_overlap,
            "top_k": args.k,
            "reset_db": bool(args.reset_db) or rag_utils.env_flag("RAG_RESET_DB", False),
        }
    )

    rag_utils.validate_env(settings)

    try:
        if args.command == "fetch":
            # Output: artifacts_dir/url/documents.json
            paths = rag_utils.artifacts_paths(settings, name="url")
            url = resolve_url(args, config)
            t0 = log_step(f"FETCH url={url}")
            try:
                docs = rag_utils.fetch_and_clean(url)
            except ValueError as e:
                if args.auto_clean:
                    rag_utils.cleanup_url_artifacts(
                        settings, name="url", clean_db=bool(args.clean_db_on_error)
                    )
                raise
            rag_utils.save_documents(docs, paths["docs"])
            log_done(t0, f"docs={len(docs)} -> {paths['docs']}")

        elif args.command == "chunk":
            # Input: artifacts_dir/url/documents.json
            # Output: artifacts_dir/url/chunks.json
            paths = rag_utils.artifacts_paths(settings, name="url")
            if not os.path.exists(paths["docs"]):
                raise FileNotFoundError(
                    f"No existe documents.json en: {paths['docs']} (ejecutá primero: fetch --url ... )"
                )

            t0 = log_step(f"CHUNK read docs -> {paths['docs']}")
            docs = rag_utils.load_documents(paths["docs"])
            try:
                rag_utils.validate_not_login_wall(docs)
            except ValueError:
                if args.auto_clean:
                    rag_utils.cleanup_url_artifacts(
                        settings, name="url", clean_db=bool(args.clean_db_on_error)
                    )
                raise
            log_done(t0, f"docs={len(docs)}")

            t1 = log_step(
                f"CHUNK split chunk_size={settings.chunk_size} overlap={settings.chunk_overlap}"
            )
            chunks = rag_utils.chunk_documents(docs, settings)
            rag_utils.save_documents(chunks, paths["chunks"])
            log_done(t1, f"chunks={len(chunks)} -> {paths['chunks']}")

        elif args.command == "index":
            # Genera embeddings + persiste en Chroma (persist_directory).
            # Si ya existe chunks.json, se reutiliza; si no, index_url se encarga del flujo completo.
            url = resolve_url(args, config)
            t0 = log_step(f"INDEX url={url}")
            try:
                rag_utils.index_url(url, settings)
            except ValueError:
                if args.auto_clean:
                    rag_utils.cleanup_url_artifacts(
                        settings, name="url", clean_db=bool(args.clean_db_on_error)
                    )
                raise
            log_done(t0, f"persist_directory={settings.persist_directory}")

        elif args.command == "ask":
            # Consulta puntual (no interactiva) sobre la BD ya indexada.
            rag_utils.ask_question(args.question, settings)
        elif args.command == "chat":
            # Modo interactivo para múltiples preguntas.
            rag_utils.interactive_mode(settings)

    except Exception as e:
        print(f"\n[ERROR] {type(e).__name__}: {e}")
        raise


if __name__ == "__main__":
    main()