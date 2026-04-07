import argparse
import json
import os
import re
import time
from typing import Any, Dict, List

from utils import rag_utils
from utils.common import load_config_file, normalize_local_path, read_text_file


DEFAULT_CONFIG_PATH = "./config/rag_config.json"
DEFAULT_OUTPUT_JSON_PATH = "./outputs/tailored_cv.json"
DEFAULT_OUTPUT_MARKDOWN_PATH = "./outputs/tailored_cv.md"
DEFAULT_PROMPT_TEMPLATE = """Tu tarea es crear un curriculum vitae en Markdown, optimizado para ATS, usando:
1) PERFIL real del candidato.
2) TODOS los jobs entregados.
3) Reporte previo de matching (si viene con contenido).

PERFIL:
{profile}

JOBS (todos):
{jobs}

REPORTE MATCHING:
{match_report}

Devuelve EXCLUSIVAMENTE este formato con etiquetas (sin texto adicional):
<TARGET_ROLE_SUMMARY>
...
</TARGET_ROLE_SUMMARY>
<CORE_REQUIREMENTS>
- ...
- ...
</CORE_REQUIREMENTS>
<COVERAGE_NOTES>
- ...
- ...
</COVERAGE_NOTES>
<ATS_KEYWORDS>
- ...
- ...
</ATS_KEYWORDS>
<CV_MARKDOWN>
...
</CV_MARKDOWN>

Reglas obligatorias:
- No inventes experiencia ni certificaciones.
- Usa solo informacion del perfil para la experiencia.
- CORE_REQUIREMENTS debe consolidar los requisitos repetidos de todos los jobs.
- COVERAGE_NOTES debe indicar cobertura o brecha principal por requisito.
- ATS_KEYWORDS debe incluir terminos tecnicos relevantes para ATS.
- CV_MARKDOWN debe incluir secciones:
  - Titulo profesional
  - Resumen
  - Habilidades clave
  - Experiencia relevante
  - Proyectos o logros
  - Educacion y certificaciones
  - Palabras clave ATS
"""


def fmt_s(seconds: float) -> str:
    return f"{seconds:.2f}s"


def log_step(label: str) -> float:
    print(f"\n[STEP] {label}")
    return time.perf_counter()


def log_done(t0: float, label: str = "done") -> float:
    dt = time.perf_counter() - t0
    print(f"    {label} ({fmt_s(dt)})")
    return dt


def resolve_job_list(job_text_glob: str) -> List[Dict[str, str]]:
    import glob

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

    raise ValueError("Falta un glob valido para jobs (ej: ./inputs/job_*.txt)")


def strip_text_fences(content: str) -> str:
    content = content.strip()
    if content.startswith("```"):
        lines = content.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        content = "\n".join(lines).strip()
    return content


def parse_tagged_response(content: str) -> Dict[str, str]:
    content = strip_text_fences(content)

    def extract(tag: str) -> str:
        pattern = rf"<{tag}>\s*(.*?)\s*</{tag}>"
        match = re.search(pattern, content, flags=re.DOTALL | re.IGNORECASE)
        if not match:
            raise ValueError(f"No se encontro la etiqueta <{tag}> en la respuesta del modelo.")
        return match.group(1).strip()

    return {
        "target_role_summary": extract("TARGET_ROLE_SUMMARY"),
        "core_requirements": extract("CORE_REQUIREMENTS"),
        "coverage_notes": extract("COVERAGE_NOTES"),
        "ats_keywords": extract("ATS_KEYWORDS"),
        "cv_markdown": extract("CV_MARKDOWN"),
    }


def extract_optional_tag(content: str, tag: str) -> str:
    pattern = rf"<{tag}>\s*(.*?)\s*</{tag}>"
    match = re.search(pattern, content, flags=re.DOTALL | re.IGNORECASE)
    if not match:
        return ""
    return match.group(1).strip()


def build_fallback_payload_from_text(content: str) -> Dict[str, Any]:
    cleaned = strip_text_fences(content)

    target_role_summary = extract_optional_tag(cleaned, "TARGET_ROLE_SUMMARY")
    core_requirements = extract_optional_tag(cleaned, "CORE_REQUIREMENTS")
    coverage_notes = extract_optional_tag(cleaned, "COVERAGE_NOTES")
    ats_keywords = extract_optional_tag(cleaned, "ATS_KEYWORDS")
    cv_markdown = extract_optional_tag(cleaned, "CV_MARKDOWN")

    if not cv_markdown:
        cv_markdown = cleaned
        cv_markdown = re.sub(
            r"<TARGET_ROLE_SUMMARY>.*?</TARGET_ROLE_SUMMARY>",
            "",
            cv_markdown,
            flags=re.DOTALL | re.IGNORECASE,
        )
        cv_markdown = re.sub(
            r"<CORE_REQUIREMENTS>.*?</CORE_REQUIREMENTS>",
            "",
            cv_markdown,
            flags=re.DOTALL | re.IGNORECASE,
        )
        cv_markdown = re.sub(
            r"<COVERAGE_NOTES>.*?</COVERAGE_NOTES>",
            "",
            cv_markdown,
            flags=re.DOTALL | re.IGNORECASE,
        )
        cv_markdown = re.sub(
            r"<ATS_KEYWORDS>.*?</ATS_KEYWORDS>",
            "",
            cv_markdown,
            flags=re.DOTALL | re.IGNORECASE,
        )
        cv_markdown = re.sub(r"</?[A-Z_]+>", "", cv_markdown, flags=re.IGNORECASE).strip()

    if not cv_markdown:
        cv_markdown = (
            "# Titulo profesional\n"
            "Perfil orientado a roles analiticos y de datos.\n\n"
            "## Resumen\n"
            "Se requiere revisar manualmente el contenido generado por el modelo.\n"
        )

    return {
        "target_role_summary": target_role_summary or "Resumen generado con formato parcial del modelo.",
        "core_requirements": core_requirements,
        "coverage_notes": coverage_notes or "No se pudo parsear completamente la salida del modelo.",
        "ats_keywords": ats_keywords,
        "cv_markdown": cv_markdown,
    }


def repair_tagged_with_llm(llm, raw_content: str) -> Dict[str, str]:
    repair_prompt = (
        "Reescribe la siguiente salida respetando SOLO estas etiquetas:\n"
        "<TARGET_ROLE_SUMMARY>...</TARGET_ROLE_SUMMARY>\n"
        "<CORE_REQUIREMENTS>...</CORE_REQUIREMENTS>\n"
        "<COVERAGE_NOTES>...</COVERAGE_NOTES>\n"
        "<ATS_KEYWORDS>...</ATS_KEYWORDS>\n"
        "<CV_MARKDOWN>...</CV_MARKDOWN>\n\n"
        "No agregues texto fuera de etiquetas.\n\n"
        "Salida original:\n"
        f"{raw_content}"
    )
    repaired = llm.invoke(repair_prompt)
    repaired_content = getattr(repaired, "content", repaired)
    if not isinstance(repaired_content, str):
        repaired_content = str(repaired_content)
    return parse_tagged_response(repaired_content)


def parse_list_block(value: Any) -> List[str]:
    if value in (None, ""):
        return []

    text = str(value).strip()
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    parsed: List[str] = []
    for line in lines:
        cleaned = line
        if cleaned.startswith("- "):
            cleaned = cleaned[2:].strip()
        elif cleaned.startswith("* "):
            cleaned = cleaned[2:].strip()
        parsed.append(cleaned)

    if parsed:
        return [item for item in parsed if item]

    return [part.strip() for part in text.split(",") if part.strip()]


def normalize_cv_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    target_role_summary = str(payload.get("target_role_summary", "")).strip()
    if not target_role_summary:
        target_role_summary = "Perfil profesional orientado a roles similares a los jobs analizados."

    cv_markdown = str(payload.get("cv_markdown", "")).strip()
    if not cv_markdown:
        raise ValueError("El modelo no devolvio CV_MARKDOWN.")

    return {
        "target_role_summary": target_role_summary,
        "core_requirements": parse_list_block(payload.get("core_requirements")),
        "coverage_notes": parse_list_block(payload.get("coverage_notes")),
        "ats_keywords": parse_list_block(payload.get("ats_keywords")),
        "cv_markdown": cv_markdown,
    }


def invoke_cv_llm(
    *,
    llm,
    prompt_template: str,
    profile_text: str,
    jobs_block: str,
    match_report_text: str,
) -> Dict[str, Any]:
    prompt = prompt_template.format(
        profile=profile_text,
        jobs=jobs_block,
        match_report=match_report_text,
    )

    t0 = log_step("llm.invoke cv")
    result = llm.invoke(prompt)
    log_done(t0)

    content = getattr(result, "content", result)
    if not isinstance(content, str):
        content = str(content)

    try:
        payload = parse_tagged_response(content)
        return normalize_cv_payload(payload)
    except Exception:
        t1 = log_step("llm.repair_tags cv")
        try:
            repaired_payload = repair_tagged_with_llm(llm, content)
            log_done(t1)
            return normalize_cv_payload(repaired_payload)
        except Exception as exc:
            log_done(t1, f"fallback ({type(exc).__name__})")
            fallback_payload = build_fallback_payload_from_text(content)
            return normalize_cv_payload(fallback_payload)


def main():
    parser = argparse.ArgumentParser(
        description="Genera un CV nuevo en base al perfil y todos los jobs."
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

    t_total = log_step("generate_cv start")

    t0 = log_step(f"load config -> {args.config}")
    config = load_config_file(args.config)
    log_done(t0, f"loaded keys={list(config.keys()) if isinstance(config, dict) else 'n/a'}")

    rag_cfg = config.get("rag", {}) if isinstance(config, dict) else {}
    match_cfg = config.get("match", {}) if isinstance(config, dict) else {}
    cv_cfg = config.get("cv", {}) if isinstance(config, dict) else {}

    t0 = log_step("build settings")
    settings = rag_utils.settings_from_dict(rag_cfg)
    if args.provider:
        settings = rag_utils.settings_from_dict({"provider": args.provider}, base=settings)
    rag_utils.validate_env(settings)
    log_done(t0, f"provider={settings.provider} ollama_base_url={settings.ollama_base_url}")

    profile_path = cv_cfg.get("profile_text_path") or match_cfg.get("profile_text_path")
    job_text_glob = cv_cfg.get("job_text_glob") or match_cfg.get("job_text_glob")
    match_report_path = cv_cfg.get("match_report_path") or match_cfg.get("output_path")
    output_json_path = cv_cfg.get("output_json_path", DEFAULT_OUTPUT_JSON_PATH)
    output_markdown_path = cv_cfg.get("output_markdown_path", DEFAULT_OUTPUT_MARKDOWN_PATH)
    prompt_template = cv_cfg.get("prompt_template", DEFAULT_PROMPT_TEMPLATE)

    if not profile_path:
        raise ValueError("Falta profile_text_path (en cv o match dentro del config).")
    if not job_text_glob:
        raise ValueError("Falta job_text_glob (en cv o match dentro del config).")

    profile_source = os.path.normpath(profile_path)
    if not os.path.exists(profile_source):
        raise FileNotFoundError(f"No existe el perfil: {profile_source}")

    t0 = log_step("init llm")
    llm = rag_utils.get_llm(settings)
    log_done(t0)

    t0 = log_step(f"read profile -> {normalize_local_path(profile_source)}")
    profile_text = read_text_file(profile_source)
    log_done(t0, f"chars={len(profile_text)}")
    if len(profile_text) < 50:
        raise ValueError("El perfil no contiene texto suficiente.")

    jobs = resolve_job_list(str(job_text_glob))
    t0 = log_step(f"read jobs -> {job_text_glob}")
    jobs_sections: List[str] = []
    for job in jobs:
        source = job["source"]
        if not (source.lower().endswith(".txt") and os.path.exists(source)):
            raise FileNotFoundError(f"No existe el job esperado: {source}")
        text = read_text_file(source)
        jobs_sections.append(f"[JOB source={job['job_source']}]\n{text}")
    jobs_block = "\n\n".join(jobs_sections).strip()
    log_done(t0, f"jobs={len(jobs)}")

    match_report_text = "No disponible."
    if isinstance(match_report_path, str) and match_report_path.strip() and os.path.exists(match_report_path):
        t0 = log_step(f"read match report -> {normalize_local_path(match_report_path)}")
        match_report_text = read_text_file(match_report_path)
        log_done(t0, f"chars={len(match_report_text)}")

    payload = invoke_cv_llm(
        llm=llm,
        prompt_template=prompt_template,
        profile_text=profile_text,
        jobs_block=jobs_block,
        match_report_text=match_report_text,
    )

    report: Dict[str, Any] = {
        "profile_source": normalize_local_path(profile_source),
        "jobs_count": len(jobs),
        "job_sources": [job["job_source"] for job in jobs],
        "match_report_path": normalize_local_path(match_report_path) if match_report_path else None,
        "target_role_summary": payload["target_role_summary"],
        "core_requirements": payload["core_requirements"],
        "coverage_notes": payload["coverage_notes"],
        "ats_keywords": payload["ats_keywords"],
    }

    t0 = log_step(f"write cv json -> {output_json_path}")
    os.makedirs(os.path.dirname(output_json_path) or ".", exist_ok=True)
    with open(output_json_path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
    log_done(t0)

    t0 = log_step(f"write cv markdown -> {output_markdown_path}")
    os.makedirs(os.path.dirname(output_markdown_path) or ".", exist_ok=True)
    with open(output_markdown_path, "w", encoding="utf-8") as handle:
        handle.write(payload["cv_markdown"].strip() + "\n")
    log_done(t0)

    print(f"\nCV generado en: {output_markdown_path}")
    print(f"Resumen estructurado en: {output_json_path}")
    log_done(t_total, "total")


if __name__ == "__main__":
    main()
