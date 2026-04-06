import os
import argparse
import json
import hashlib
import time
import glob
import math
from typing import Dict, Any, List

import rag_utils
from utils.common import load_config_file, read_text_file
from langchain_core.documents import Document


DEFAULT_CONFIG_PATH = "./config/rag_config.json"


def env_ollama_base_url_present() -> bool:
    return os.getenv("OLLAMA_BASE_URL") is not None or os.getenv("RAG_OLLAMA_BASE_URL") is not None


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


def resolve_profile_text(match_cfg: dict, settings: rag_utils.RAGSettings, refresh: bool) -> str:
    profile_text = match_cfg.get("profile_text")
    if isinstance(profile_text, str) and profile_text.strip():
        t0 = log_step("profile source=inline")
        log_done(t0)
        return profile_text.strip()

    profile_text_path = match_cfg.get("profile_text_path")
    if isinstance(profile_text_path, str) and profile_text_path.strip():
        t0 = log_step(f"profile source=file -> {profile_text_path}")
        text = read_text_file(profile_text_path)
        log_done(t0, f"read chars={len(text)}")
        return text

    profile_url = match_cfg.get("profile_url")
    if not profile_url:
        raise ValueError(
            "Falta match.profile_url (o alternativa match.profile_text / match.profile_text_path) en rag_config.json"
        )

    return load_or_fetch_text(profile_url, settings, dataset_prefix="profile", refresh=refresh)


def resolve_job_text(match_cfg: dict, job_url: str, settings: rag_utils.RAGSettings, refresh: bool) -> str:
    job_texts = match_cfg.get("job_texts")
    if isinstance(job_texts, dict):
        inline = job_texts.get(job_url)
        if isinstance(inline, str) and inline.strip():
            t0 = log_step("job source=inline")
            log_done(t0)
            return inline.strip()

    job_text_paths = match_cfg.get("job_text_paths")
    if isinstance(job_text_paths, dict):
        p = job_text_paths.get(job_url)
        if isinstance(p, str) and p.strip():
            t0 = log_step(f"job source=file -> {p}")
            text = read_text_file(p)
            log_done(t0, f"read chars={len(text)}")
            return text

    return load_or_fetch_text(job_url, settings, dataset_prefix="job", refresh=refresh)


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


def invoke_match_llm(
    llm,
    profile_text: str,
    job_text: str,
    job_url: str,
    prompt_template: str,
) -> Dict[str, Any]:
    prompt = prompt_template.format(profile=profile_text, job=job_text, job_url=job_url)

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
    parser = argparse.ArgumentParser(
        description="Evalúa compatibilidad entre un perfil de LinkedIn y una lista de jobs (score + razones)."
    )
    parser.add_argument(
        "--config",
        default=DEFAULT_CONFIG_PATH,
        help="Archivo de configuración JSON (por defecto: ./config/rag_config.json)",
    )
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="Fuerza re-descargar perfil/jobs (ignora cache en rag_artifacts)",
    )

    parser.add_argument(
        "--provider",
        choices=["openai", "ollama"],
        default=None,
        help="Sobrescribe el provider del JSON (openai u ollama).",
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
    if env_ollama_base_url_present() and settings.provider.lower() == "ollama":
        settings = rag_utils.settings_from_dict(
            {"ollama_base_url": rag_utils.DEFAULT_OLLAMA_BASE_URL},
            base=settings,
        )

    if args.provider:
        settings = rag_utils.settings_from_dict({"provider": args.provider}, base=settings)
    rag_utils.validate_env(settings)
    log_done(t0, f"provider={settings.provider} ollama_base_url={settings.ollama_base_url}")

    profile_source = match_cfg.get("profile_text_path") or match_cfg.get("profile_text") or "inline"
    jobs = resolve_job_list(match_cfg)
    output_path = match_cfg.get("output_path", "./outputs/job_match_report.json")
    prompt_template = match_cfg.get("prompt_template")

    if not (match_cfg.get("profile_text") or match_cfg.get("profile_text_path")):
        raise ValueError(
            "Falta match.profile_text_path (o alternativa match.profile_text) en rag_config.json"
        )
    if not jobs:
        raise ValueError(
            "Faltan jobs. Configurá match.job_text_glob (ej: ./inputs/job_*.txt) o match.job_text_files o match.job_urls."
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
    if match_cfg.get("profile_text_path"):
        print(f"    source=file: {match_cfg.get('profile_text_path')}")
    elif match_cfg.get("profile_text"):
        print("    source=inline")
    else:
        print(f"    source=unknown")

    t0 = time.perf_counter()
    profile_text = resolve_profile_text(match_cfg, settings, refresh=bool(args.refresh))
    print(f"    profile loaded chars={len(profile_text)} ({fmt_s(time.perf_counter() - t0)})")
    if len(profile_text) < 50:
        raise ValueError("El perfil no devolvió suficiente texto. Probablemente LinkedIn bloqueó/redirect.")

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
                job_text = resolve_job_text(match_cfg, source, settings, refresh=bool(args.refresh))
            print(f"    job loaded chars={len(job_text)} ({fmt_s(time.perf_counter() - t0)})")
            if len(job_text) < 50:
                raise ValueError("Job sin texto útil (bloqueo/login/JS)")

            if hasattr(rag_utils, "is_probably_linkedin_login_wall") and rag_utils.is_probably_linkedin_login_wall(job_text):
                raise ValueError("Job parece ser login wall/landing (texto no útil)")

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
        "profile_url": profile_url,
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
