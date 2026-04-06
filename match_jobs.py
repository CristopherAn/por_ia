import os
import argparse
import json
import math
from typing import Dict, Any, List

import rag_utils
from utils.common import load_config_file, read_text_file
from langchain_core.documents import Document


DEFAULT_CONFIG_PATH = "./config/rag_config.json"


def fmt_s(seconds: float) -> str:
    return f"{seconds:.2f}s"


def log_step(label: str) -> float:
    import time
    print(f"\n[STEP] {label}")
    return time.perf_counter()


def log_done(t0: float, label: str = "done") -> float:
    import time
    dt = time.perf_counter() - t0
    print(f"    {label} ({fmt_s(dt)})")
    return dt


def resolve_profile_text(profile_text_path: str) -> str:
    """Carga el perfil desde un archivo local."""
    if not profile_text_path or not profile_text_path.strip():
        raise ValueError("Falta match.profile_text_path en rag_config.json")
    
    if not os.path.exists(profile_text_path):
        raise FileNotFoundError(f"No existe el archivo de perfil: {profile_text_path}")
    
    t0 = log_step(f"profile source=file -> {profile_text_path}")
    text = read_text_file(profile_text_path)
    log_done(t0, f"read chars={len(text)}")
    return text


def resolve_job_list(match_cfg: dict) -> List[Dict[str, str]]:
    """Resuelve la lista de jobs desde glob pattern."""
    import glob
    
    job_text_glob = match_cfg.get("job_text_glob")
    if isinstance(job_text_glob, str) and job_text_glob.strip():
        paths = sorted(glob.glob(job_text_glob))
        if paths:
            return [{"job_id": str(p), "source": str(p)} for p in paths]
    
    raise ValueError("Falta match.job_text_glob en rag_config.json")


def invoke_match_llm(
    llm,
    profile_text: str,
    job_text: str,
    job_source: str,
    prompt_template: str,
) -> Dict[str, Any]:
    prompt = prompt_template.format(profile=profile_text, job=job_text, job_source=job_source)

    t0 = log_step("llm.invoke")
    result = llm.invoke(prompt)
    log_done(t0)

    content = getattr(result, "content", result)
    if not isinstance(content, str):
        content = str(content)

    content = content.strip()
    start = content.find("{")
    end = content.rfind("}")
    if start != -1 and end != -1 and end > start:
        content = content[start : end + 1]

    return json.loads(content)


def cosine_similarity(a: List[float], b: List[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = 0.0
    na = 0.0
    nb = 0.0
    for x, y in zip(a, b):
        dot += x * y
        na += x * x
        nb += y * y
    denom = math.sqrt(na) * math.sqrt(nb)
    if denom == 0.0:
        return 0.0
    return dot / denom


def retrieve_relevant_text(
    *,
    embeddings,
    query_text: str,
    candidate_text: str,
    settings: rag_utils.RAGSettings,
    top_k_chunks: int,
    label: str,
) -> str:
    t0 = log_step(f"chunk -> {label}")
    candidate_docs = rag_utils.split_documents(
        [Document(page_content=candidate_text, metadata={})], settings
    )
    log_done(t0, f"chunks={len(candidate_docs)}")

    if not candidate_docs:
        return candidate_text

    t1 = log_step(f"embeddings -> {label} (chunks)")
    chunk_vectors = embeddings.embed_documents([d.page_content for d in candidate_docs])
    log_done(t1)

    t2 = log_step(f"embeddings -> {label} (query)")
    query_vec = embeddings.embed_query(query_text)
    log_done(t2)

    scored = []
    for i, vec in enumerate(chunk_vectors):
        scored.append((cosine_similarity(query_vec, vec), i))
    scored.sort(reverse=True, key=lambda x: x[0])

    k = max(1, min(int(top_k_chunks), len(candidate_docs)))
    top = scored[:k]
    idxs = [i for _, i in top]
    selected = [candidate_docs[i].page_content for i in idxs]

    t3 = log_step(f"retrieve -> {label}")
    log_done(t3, f"selected_chunks={len(selected)}")

    return "\n\n".join(selected).strip()


def main():
    import time
    
    parser = argparse.ArgumentParser(
        description="Evalúa compatibilidad entre un perfil y una lista de jobs (score + razones)."
    )
    parser.add_argument(
        "--config",
        default=DEFAULT_CONFIG_PATH,
        help="Archivo de configuración JSON (por defecto: ./config/rag_config.json)",
    )
    parser.add_argument(
        "--provider",
        choices=["ollama"],
        default=None,
        help="Sobrescribe el provider del JSON (solo ollama).",
    )
    args = parser.parse_args()

    t_total = log_step("match_linkedin_jobs start")

    fallback = "./rag_config.json" if args.config == DEFAULT_CONFIG_PATH else None
    t0 = log_step(f"load config -> {args.config}")
    config = load_config_file(args.config, fallback_path=fallback)
    log_done(t0, f"loaded keys={list(config.keys()) if isinstance(config, dict) else 'n/a'}")
    rag_cfg = config.get("rag", {}) if isinstance(config, dict) else {}
    match_cfg = config.get("match", {}) if isinstance(config, dict) else {}
    retrieval_top_k_chunks = int(match_cfg.get("retrieval_top_k_chunks", 6))

    t0 = log_step("build settings")
    settings = rag_utils.settings_from_dict(rag_cfg)
    if args.provider:
        settings = rag_utils.settings_from_dict({"provider": args.provider}, base=settings)
    rag_utils.validate_env(settings)
    log_done(t0, f"provider={settings.provider} ollama_base_url={settings.ollama_base_url}")

    profile_source = match_cfg.get("profile_text_path", "default")
    jobs = resolve_job_list(match_cfg)
    output_path = match_cfg.get("output_path", "./outputs/job_match_report.json")
    prompt_template = match_cfg.get("prompt_template")

    if not match_cfg.get("profile_text_path"):
        raise ValueError(
            "Falta match.profile_text_path en rag_config.json"
        )
    if not jobs:
        raise ValueError(
            "Faltan jobs. Configurá match.job_text_glob (ej: ./inputs/job_*.txt)."
        )
    if not prompt_template:
        raise ValueError("Falta match.prompt_template en rag_config.json")

    t0 = log_step("init llm")
    llm = rag_utils.get_llm(settings)
    log_done(t0)

    t0 = log_step("init embeddings")
    embeddings = rag_utils.get_embeddings(settings)
    log_done(t0)

    print("\n[1/2] Cargando perfil")
    print(f"    source=file: {match_cfg.get('profile_text_path')}")
    t0 = time.perf_counter()
    profile_text = resolve_profile_text(match_cfg.get('profile_text_path'))
    print(f"    profile loaded chars={len(profile_text)} ({fmt_s(time.perf_counter() - t0)})")
    if len(profile_text) < 50:
        raise ValueError("El perfil no devolvió suficiente texto.")

    results: List[Dict[str, Any]] = []
    errors: List[Dict[str, Any]] = []

    print(f"\n[2/2] Evaluando jobs: {len(jobs)}")
    for i, job_item in enumerate(jobs, start=1):
        t_job = time.perf_counter()
        try:
            job_id = job_item["job_id"]
            source = job_item["source"]
            print(f"\n  [{i}/{len(jobs)}] Job: {source}")
            t0 = time.perf_counter()
            if source.lower().endswith(".txt") and os.path.exists(source):
                job_text = read_text_file(source)
            else:
                raise ValueError(f"Solo se soportan archivos .txt locales: {source}")
            print(f"    job loaded chars={len(job_text)} ({fmt_s(time.perf_counter() - t0)})")
            if len(job_text) < 50:
                raise ValueError("Job sin texto útil")

            t0 = log_step("retrieve context (profile<->job)")
            job_context = retrieve_relevant_text(
                embeddings=embeddings,
                query_text=profile_text,
                candidate_text=job_text,
                settings=settings,
                top_k_chunks=retrieval_top_k_chunks,
                label="job",
            )
            profile_context = retrieve_relevant_text(
                embeddings=embeddings,
                query_text=job_text,
                candidate_text=profile_text,
                settings=settings,
                top_k_chunks=retrieval_top_k_chunks,
                label="profile",
            )
            log_done(t0)

            match = invoke_match_llm(
                llm=llm,
                profile_text=profile_context,
                job_text=job_context,
                job_source=source,
                prompt_template=prompt_template,
            )
            results.append(match)
            print(f"    OK total_job_time={fmt_s(time.perf_counter() - t_job)}")
        except Exception as e:
            errors.append({"job_source": job_item.get("source"), "error": f"{type(e).__name__}: {e}"})
            print(f"    ERROR total_job_time={fmt_s(time.perf_counter() - t_job)} -> {type(e).__name__}: {e}")

    report: Dict[str, Any] = {
        "profile_source": profile_source,
        "jobs_count": len(jobs),
        "results": results,
        "errors": errors,
        "provider": settings.provider,
        "ollama_base_url": settings.ollama_base_url,
        "ollama_chat_model": settings.ollama_chat_model,
    }

    t0 = log_step(f"write report -> {output_path}")
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    log_done(t0)

    print(f"\nReporte generado en: {output_path}")
    if errors:
        print(f"\n[WARN] Hubo errores en {len(errors)} jobs. Revisá la sección errors del reporte.")

    log_done(t_total, "total")


if __name__ == "__main__":
    main()
