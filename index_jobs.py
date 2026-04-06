import os
import argparse
import glob
import time
from typing import Dict, Any, List

import rag_utils
from utils.common import load_config_file, read_text_file


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
    """Resuelve la lista de jobs desde glob pattern."""
    job_text_glob = match_cfg.get("job_text_glob")
    if isinstance(job_text_glob, str) and job_text_glob.strip():
        paths = sorted(glob.glob(job_text_glob))
        if paths:
            return [{"job_id": str(p), "source": str(p)} for p in paths]
    
    raise ValueError("Falta match.job_text_glob en rag_config.json")


def main():
    parser = argparse.ArgumentParser(description="Indexa jobs en ChromaDB (etapa de 'entrenamiento'). Se crea una nueva indexación cada vez.")
    parser.add_argument("--config", default=DEFAULT_CONFIG_PATH, help="Archivo de configuración JSON")

    args = parser.parse_args()
    config = load_config_file(args.config)
    rag_cfg = config.get("rag", {})
    match_cfg = config.get("match", {})
    
    settings = rag_utils.settings_from_dict(rag_cfg)
    
    # Siempre resetear la BD anterior: cada nueva indexación borra la antigua
    settings = rag_utils.settings_from_dict({"reset_db": True}, base=settings)

    rag_utils.validate_env(settings)

    t0 = log_step("init embeddings")
    embeddings = rag_utils.get_embeddings(settings)
    log_done(t0)

    jobs = resolve_job_list(match_cfg)
    if not jobs:
        raise ValueError("No jobs encontrados. Configura match.job_text_glob en rag_config.json (ej: ./inputs/job_*.txt)")

    print(f"\n[INDEXING] Procesando {len(jobs)} jobs")
    print(f"    Nota: Se elimina la BD anterior en: {settings.persist_directory}")

    all_chunks = []
    for i, job_item in enumerate(jobs, start=1):
        source = job_item["source"]
        print(f"\n  [{i}/{len(jobs)}] Job: {source}")

        if not (source.lower().endswith(".txt") and os.path.exists(source)):
            raise FileNotFoundError(f"Solo se soportan archivos .txt locales: {source}")
        
        text = read_text_file(source)
        job_id = job_item["job_id"]

        t0 = time.perf_counter()
        docs = rag_utils.split_documents(
            [rag_utils.Document(page_content=text, metadata={"job_id": job_id, "source": source})],
            settings
        )
        all_chunks.extend(docs)
        print(f"    chunked {len(docs)} chunks ({fmt_s(time.perf_counter() - t0)})")

    t0 = log_step("index chunks in ChromaDB")
    rag_utils.index_chunks(all_chunks, settings)
    log_done(t0, f"indexed {len(all_chunks)} chunks")

    print(f"\nIndexing completado. Base de datos en: {settings.persist_directory}")


if __name__ == "__main__":
    main()