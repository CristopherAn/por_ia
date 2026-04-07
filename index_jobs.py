import argparse
import glob
import os
import time
from typing import Dict, List

from utils import rag_utils
from utils.common import load_config_file, normalize_local_path, read_text_file


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


def resolve_job_list(match_cfg: dict) -> List[Dict[str, str]]:
    job_text_glob = match_cfg.get("job_text_glob")
    if isinstance(job_text_glob, str) and job_text_glob.strip():
        paths = sorted(glob.glob(job_text_glob))
        if paths:
            return [
                {
                    "job_id": normalize_local_path(path),
                    "source": os.path.normpath(path),
                    "job_source": normalize_local_path(path),
                }
                for path in paths
            ]

    raise ValueError("Falta match.job_text_glob en rag_config.json")


def main():
    parser = argparse.ArgumentParser(
        description="Indexa jobs en ChromaDB. Cada nueva indexacion reemplaza la anterior."
    )
    parser.add_argument("--config", default=DEFAULT_CONFIG_PATH, help="Archivo de configuracion JSON")
    parser.add_argument(
        "--provider",
        choices=["ollama"],
        default=None,
        help="Sobrescribe el provider del JSON.",
    )
    parser.add_argument(
        "--chat-model",
        default=None,
        help="Sobrescribe rag.ollama_chat_model para la ejecucion.",
    )
    parser.add_argument(
        "--embedding-model",
        default=None,
        help="Sobrescribe rag.ollama_embedding_model para la ejecucion.",
    )
    parser.add_argument(
        "--ollama-base-url",
        default=None,
        help="Sobrescribe rag.ollama_base_url para la ejecucion.",
    )
    args = parser.parse_args()

    config = load_config_file(args.config)
    rag_cfg = config.get("rag", {})
    match_cfg = config.get("match", {})

    settings = rag_utils.settings_from_dict(rag_cfg)
    runtime_overrides = {"reset_db": True}
    if args.provider:
        runtime_overrides["provider"] = args.provider
    if args.chat_model:
        runtime_overrides["ollama_chat_model"] = args.chat_model
    if args.embedding_model:
        runtime_overrides["ollama_embedding_model"] = args.embedding_model
    if args.ollama_base_url:
        runtime_overrides["ollama_base_url"] = args.ollama_base_url
    settings = rag_utils.settings_from_dict(runtime_overrides, base=settings)
    rag_utils.validate_env(settings)

    t0 = log_step("init embeddings")
    rag_utils.get_embeddings(settings)
    log_done(
        t0,
        (
            f"provider={settings.provider} "
            f"base_url={settings.ollama_base_url} "
            f"embedding_model={settings.ollama_embedding_model}"
        ),
    )

    jobs = resolve_job_list(match_cfg)
    if not jobs:
        raise ValueError("No jobs encontrados. Configura match.job_text_glob en rag_config.json.")

    print(f"\n[INDEXING] Procesando {len(jobs)} jobs")
    print(f"    Nota: Se elimina la BD anterior en: {settings.persist_directory}")

    all_chunks = []
    for i, job_item in enumerate(jobs, start=1):
        source = job_item["source"]
        job_source = job_item["job_source"]
        job_id = job_item["job_id"]
        print(f"\n  [{i}/{len(jobs)}] Job: {job_source}")

        if not (source.lower().endswith(".txt") and os.path.exists(source)):
            raise FileNotFoundError(f"Solo se soportan archivos .txt locales: {source}")

        text = read_text_file(source)
        t0 = time.perf_counter()
        docs = rag_utils.split_documents(
            [rag_utils.Document(page_content=text, metadata={"job_id": job_id, "source": job_source})],
            settings,
        )
        all_chunks.extend(docs)
        print(f"    chunked {len(docs)} chunks ({fmt_s(time.perf_counter() - t0)})")

    t0 = log_step("index chunks in ChromaDB")
    persist_used = rag_utils.index_chunks(all_chunks, settings)
    log_done(t0, f"indexed {len(all_chunks)} chunks")

    print(f"\nIndexing completado. Base de datos en: {persist_used}")
    if os.path.normpath(persist_used) != os.path.normpath(settings.persist_directory):
        print(
            "    [WARN] Se uso una ruta fallback. "
            "Quedo registrada y se reutiliza automaticamente en matching/cv."
        )


if __name__ == "__main__":
    main()
