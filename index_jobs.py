import os
import argparse
import json
import hashlib
import time
import glob
from typing import Dict, Any, List

import rag_utils
from utils.common import load_config_file


DEFAULT_CONFIG_PATH = "./config/rag_config.json"


def url_id(url: str) -> str:
    return hashlib.sha256(url.strip().encode("utf-8")).hexdigest()[:16]


def dataset_name(prefix: str, url: str) -> str:
    return f"{prefix}_{url_id(url)}"


def fmt_s(seconds: float) -> str:
    return f"{seconds:.2f}s"


def log_step(label: str) -> float:
    print(f"\n[STEP] {label}")
    return time.perf_counter()


def log_done(t0: float, label: str = "done") -> float:
    dt = time.perf_counter() - t0
    print(f"    {label} ({fmt_s(dt)})")
    return dt


def load_or_fetch_text(url: str, settings: rag_utils.RAGSettings, dataset_prefix: str, refresh: bool) -> str:
    name = dataset_name(dataset_prefix, url)
    paths = rag_utils.artifacts_paths(settings, name=name)

    if (not refresh) and os.path.exists(paths["docs"]):
        t0 = log_step(f"cache hit -> {dataset_prefix} docs ({name})")
        docs = rag_utils.load_documents(paths["docs"])
        log_done(t0, f"loaded docs={len(docs)}")
    else:
        t0 = log_step(f"fetch -> {dataset_prefix} url ({name})")
        docs = rag_utils.fetch_and_clean(url)
        log_done(t0, f"fetched+cleaned docs={len(docs)}")

        t1 = log_step(f"save -> {dataset_prefix} docs cache ({name})")
        rag_utils.save_documents(docs, paths["docs"])
        log_done(t1, f"saved -> {paths['docs']}")

    return "\n\n".join([d.page_content for d in docs]).strip()


def resolve_job_list(match_cfg: dict) -> List[Dict[str, str]]:
    job_text_files = match_cfg.get("job_text_files")
    if isinstance(job_text_files, list) and job_text_files:
        return [{"job_id": str(p), "source": str(p)} for p in job_text_files if str(p).strip()]

    job_text_glob = match_cfg.get("job_text_glob")
    if isinstance(job_text_glob, str) and job_text_glob.strip():
        paths = sorted(glob.glob(job_text_glob))
        return [{"job_id": str(p), "source": str(p)} for p in paths]

    job_urls = match_cfg.get("job_urls") or []
    if isinstance(job_urls, list) and job_urls:
        return [{"job_id": str(u), "source": str(u)} for u in job_urls if str(u).strip()]

    return []


def main():
    parser = argparse.ArgumentParser(description="Indexa jobs en ChromaDB (etapa de 'entrenamiento').")
    parser.add_argument("--config", default=DEFAULT_CONFIG_PATH, help="Archivo de configuración JSON")
    parser.add_argument("--refresh", action="store_true", help="Forzar refresh de cache de URLs")
    parser.add_argument("--reset-db", action="store_true", help="Resetear la base de datos vectorial")

    args = parser.parse_args()
    settings = rag_utils.settings_from_dict(load_config_file(args.config, {}).get("rag", {}))
    if args.reset_db:
        settings = rag_utils.settings_from_dict({"reset_db": True}, base=settings)

    rag_utils.validate_env(settings)

    t0 = log_step("init embeddings")
    embeddings = rag_utils.get_embeddings(settings)
    log_done(t0)

    match_cfg = load_config_file(args.config, {}).get("match", {})
    jobs = resolve_job_list(match_cfg)

    if not jobs:
        raise ValueError("No jobs encontrados. Configura match.job_text_glob o match.job_text_files.")

    print(f"\n[INDEXING] Procesando {len(jobs)} jobs")

    all_chunks = []
    for i, job_item in enumerate(jobs, start=1):
        job_id = job_item["job_id"]
        source = job_item["source"]
        print(f"\n  [{i}/{len(jobs)}] Job: {source}")

        if source.lower().endswith(".txt") and os.path.exists(source):
            text = rag_utils.read_text_file(source)
        else:
            text = load_or_fetch_text(source, settings, dataset_prefix="job", refresh=bool(args.refresh))

        t0 = time.perf_counter()
        docs = rag_utils.split_documents([rag_utils.Document(page_content=text, metadata={"job_id": job_id, "source": source})], settings)
        all_chunks.extend(docs)
        print(f"    chunked {len(docs)} chunks ({fmt_s(time.perf_counter() - t0)})")

    t0 = log_step("index chunks in ChromaDB")
    rag_utils.index_chunks(all_chunks, settings)
    log_done(t0, f"indexed {len(all_chunks)} chunks")

    print(f"\nIndexing completado. Base de datos en: {settings.persist_directory}")


if __name__ == "__main__":
    main()