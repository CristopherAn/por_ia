import argparse
import json
import math
import os
import time
from json import JSONDecodeError
from typing import Any, Dict, List, Tuple

from langchain_core.documents import Document

import rag_utils
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


def resolve_profile_text(profile_text_path: str) -> str:
    if not profile_text_path or not profile_text_path.strip():
        raise ValueError("Falta match.profile_text_path en rag_config.json")

    source = os.path.normpath(profile_text_path)
    if not os.path.exists(source):
        raise FileNotFoundError(f"No existe el archivo de perfil: {source}")

    t0 = log_step(f"profile source=file -> {normalize_local_path(source)}")
    text = read_text_file(source)
    log_done(t0, f"read chars={len(text)}")
    return text


def resolve_job_list(match_cfg: dict) -> List[Dict[str, str]]:
    import glob

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


def strip_json_fences(content: str) -> str:
    content = content.strip()
    if content.startswith("```"):
        lines = content.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        content = "\n".join(lines).strip()
    return content


def parse_json_response(content: str) -> Dict[str, Any]:
    content = strip_json_fences(content)
    decoder = json.JSONDecoder()

    candidates = [content]
    if "{" in content:
        candidates.append(content[content.find("{") :])

    for candidate in candidates:
        candidate = candidate.strip()
        if not candidate:
            continue
        try:
            data, _ = decoder.raw_decode(candidate)
        except JSONDecodeError:
            continue
        if isinstance(data, dict):
            return data

    start = content.find("{")
    end = content.rfind("}")
    if start != -1 and end != -1 and end > start:
        return json.loads(content[start : end + 1])

    raise JSONDecodeError("No se encontro un objeto JSON valido en la respuesta del modelo", content, 0)


def repair_json_with_llm(llm, raw_content: str, job_source: str) -> Dict[str, Any]:
    repair_prompt = (
        "Convierte la siguiente salida en un JSON valido. "
        "Responde solo con un objeto JSON, sin markdown ni explicaciones.\n\n"
        f"job_source esperado: {job_source}\n\n"
        "Salida original:\n"
        f"{raw_content}"
    )
    repaired = llm.invoke(repair_prompt)
    repaired_content = getattr(repaired, "content", repaired)
    if not isinstance(repaired_content, str):
        repaired_content = str(repaired_content)
    return parse_json_response(repaired_content)


def normalize_match_payload(payload: Dict[str, Any], job_source: str) -> Dict[str, Any]:
    def as_list(value: Any) -> List[str]:
        if isinstance(value, list):
            return [str(item).strip() for item in value if str(item).strip()]
        if value in (None, ""):
            return []
        return [str(value).strip()]

    fit_score = payload.get("fit_score", 0)
    try:
        fit_score = int(fit_score)
    except (TypeError, ValueError):
        fit_score = 0
    fit_score = max(0, min(100, fit_score))

    seniority = str(payload.get("seniority_guess", "unknown")).strip().lower() or "unknown"
    if seniority not in {"junior", "semi-senior", "senior", "lead", "unknown"}:
        seniority = "unknown"

    recommendation = str(payload.get("final_recommendation", "maybe")).strip().lower() or "maybe"
    if recommendation not in {"apply", "maybe", "no"}:
        recommendation = "maybe"

    return {
        "job_source": job_source,
        "fit_score": fit_score,
        "seniority_guess": seniority,
        "pros": as_list(payload.get("pros")),
        "gaps": as_list(payload.get("gaps")),
        "missing_keywords": as_list(payload.get("missing_keywords")),
        "recommended_cv_bullets": as_list(payload.get("recommended_cv_bullets")),
        "final_recommendation": recommendation,
    }


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

    try:
        payload = parse_json_response(content)
    except JSONDecodeError:
        t1 = log_step("llm.repair_json")
        payload = repair_json_with_llm(llm, content, job_source)
        log_done(t1)

    return normalize_match_payload(payload, job_source)


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
    chunk_vectors = embeddings.embed_documents([doc.page_content for doc in candidate_docs])
    log_done(t1)

    t2 = log_step(f"embeddings -> {label} (query)")
    query_vec = embeddings.embed_query(query_text)
    log_done(t2)

    scored = [
        (cosine_similarity(query_vec, vector), idx)
        for idx, vector in enumerate(chunk_vectors)
    ]
    scored.sort(reverse=True, key=lambda item: item[0])

    k = max(1, min(int(top_k_chunks), len(candidate_docs)))
    selected = [candidate_docs[idx].page_content for _, idx in scored[:k]]

    t3 = log_step(f"retrieve -> {label}")
    log_done(t3, f"selected_chunks={len(selected)}")
    return "\n\n".join(selected).strip()


def retrieve_job_context_from_index(
    *,
    vectorstore,
    query_text: str,
    job_id: str,
    fallback_text: str,
    top_k_chunks: int,
) -> Tuple[str, bool]:
    t0 = log_step("retrieve indexed job context")
    docs = vectorstore.similarity_search(
        query_text,
        k=max(1, int(top_k_chunks)),
        filter={"job_id": job_id},
    )
    log_done(t0, f"selected_chunks={len(docs)}")

    if not docs:
        return fallback_text, False

    return "\n\n".join(doc.page_content for doc in docs).strip(), True


def main():
    parser = argparse.ArgumentParser(
        description="Evalua compatibilidad entre un perfil y una lista de jobs."
    )
    parser.add_argument(
        "--config",
        default=DEFAULT_CONFIG_PATH,
        help="Archivo de configuracion JSON (por defecto: ./config/rag_config.json)",
    )
    parser.add_argument(
        "--provider",
        choices=["ollama"],
        default=None,
        help="Sobrescribe el provider del JSON.",
    )
    args = parser.parse_args()

    t_total = log_step("match_jobs start")

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
        raise ValueError("Falta match.profile_text_path en rag_config.json")
    if not jobs:
        raise ValueError("Faltan jobs. Configura match.job_text_glob (ej: ./inputs/job_*.txt).")
    if not prompt_template:
        raise ValueError("Falta match.prompt_template en rag_config.json")

    t0 = log_step("init llm")
    llm = rag_utils.get_llm(settings)
    log_done(t0)

    t0 = log_step("init embeddings")
    embeddings = rag_utils.get_embeddings(settings)
    log_done(t0)

    t0 = log_step("load vectorstore")
    vectorstore = rag_utils.load_vectorstore(settings)
    log_done(t0, f"collection={settings.collection_name}")

    print("\n[1/2] Cargando perfil")
    print(f"    source=file: {normalize_local_path(profile_source)}")
    t0 = time.perf_counter()
    profile_text = resolve_profile_text(profile_source)
    print(f"    profile loaded chars={len(profile_text)} ({fmt_s(time.perf_counter() - t0)})")
    if len(profile_text) < 50:
        raise ValueError("El perfil no devolvio suficiente texto.")

    results: List[Dict[str, Any]] = []
    errors: List[Dict[str, Any]] = []

    print(f"\n[2/2] Evaluando jobs: {len(jobs)}")
    for i, job_item in enumerate(jobs, start=1):
        t_job = time.perf_counter()
        try:
            job_id = job_item["job_id"]
            source = job_item["source"]
            job_source = job_item["job_source"]
            print(f"\n  [{i}/{len(jobs)}] Job: {job_source}")

            t0 = time.perf_counter()
            if source.lower().endswith(".txt") and os.path.exists(source):
                job_text = read_text_file(source)
            else:
                raise ValueError(f"Solo se soportan archivos .txt locales: {source}")
            print(f"    job loaded chars={len(job_text)} ({fmt_s(time.perf_counter() - t0)})")
            if len(job_text) < 50:
                raise ValueError("Job sin texto util")

            t0 = log_step("retrieve context (profile<->job)")
            job_context, used_index = retrieve_job_context_from_index(
                vectorstore=vectorstore,
                query_text=profile_text,
                job_id=job_id,
                fallback_text=job_text,
                top_k_chunks=retrieval_top_k_chunks,
            )
            profile_context = retrieve_relevant_text(
                embeddings=embeddings,
                query_text=job_text,
                candidate_text=profile_text,
                settings=settings,
                top_k_chunks=retrieval_top_k_chunks,
                label="profile",
            )
            log_done(t0, f"job_source={'index' if used_index else 'fallback'}")

            match = invoke_match_llm(
                llm=llm,
                profile_text=profile_context,
                job_text=job_context,
                job_source=job_source,
                prompt_template=prompt_template,
            )
            results.append(match)
            print(f"    OK total_job_time={fmt_s(time.perf_counter() - t_job)}")
        except Exception as exc:
            errors.append(
                {
                    "job_source": job_item.get("job_source"),
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
            print(f"    ERROR total_job_time={fmt_s(time.perf_counter() - t_job)} -> {type(exc).__name__}: {exc}")

    report: Dict[str, Any] = {
        "profile_source": normalize_local_path(profile_source),
        "jobs_count": len(jobs),
        "results": results,
        "errors": errors,
        "provider": settings.provider,
        "ollama_base_url": settings.ollama_base_url,
        "ollama_chat_model": settings.ollama_chat_model,
    }

    t0 = log_step(f"write report -> {output_path}")
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
    log_done(t0)

    print(f"\nReporte generado en: {output_path}")
    if errors:
        print(f"\n[WARN] Hubo errores en {len(errors)} jobs. Revisa la seccion errors del reporte.")

    log_done(t_total, "total")


if __name__ == "__main__":
    main()
