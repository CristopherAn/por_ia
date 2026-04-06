import argparse
import json
import os
from collections import Counter
from datetime import datetime
from typing import Any, Dict, List, Tuple


def load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f) or {}


def clamp_int(x: Any, default: int = 0) -> int:
    try:
        v = int(x)
    except Exception:
        return default
    return max(0, min(100, v))


def norm_list(x: Any) -> List[str]:
    if not isinstance(x, list):
        return []
    out: List[str] = []
    for item in x:
        if isinstance(item, str) and item.strip():
            out.append(item.strip())
    return out


def job_sort_key(item: Dict[str, Any]) -> Tuple[int, str]:
    return (clamp_int(item.get("fit_score"), 0), str(item.get("job_url") or ""))


def top_common(counter: Counter, n: int) -> List[Tuple[str, int]]:
    return [(k, v) for k, v in counter.most_common(n) if k]


def summarize_actions(job: Dict[str, Any]) -> List[str]:
    score = clamp_int(job.get("fit_score"), 0)
    gaps = norm_list(job.get("gaps"))
    missing = norm_list(job.get("missing_keywords"))

    actions: List[str] = []

    if score >= 80:
        actions.append("Postular esta semana y adaptar CV al rol (1 página, bullets con impacto).")
    elif score >= 65:
        actions.append("Postular si el rol encaja con tus objetivos; mejorar keywords y bullets del CV antes.")
    else:
        actions.append("Solo postular si hay alta motivación; priorizar cerrar gaps o elegir roles más alineados.")

    if gaps:
        actions.append("Cubrir gaps: preparar evidencia/proyectos concretos para responderlos en entrevista.")
    if missing:
        actions.append("Actualizar CV/LinkedIn incorporando keywords faltantes (si aplican realmente).")

    return actions


def md_escape(s: str) -> str:
    return s.replace("|", "\\|")


def render_table(rows: List[List[str]]) -> str:
    if not rows:
        return ""
    header = rows[0]
    out = []
    out.append("| " + " | ".join(header) + " |")
    out.append("| " + " | ".join(["---"] * len(header)) + " |")
    for r in rows[1:]:
        out.append("| " + " | ".join(r) + " |")
    return "\n".join(out)


def generate_markdown(report: Dict[str, Any], top_common_n: int = 10) -> str:
    results = report.get("results") if isinstance(report, dict) else []
    if not isinstance(results, list):
        results = []

    jobs: List[Dict[str, Any]] = [j for j in results if isinstance(j, dict)]
    jobs_sorted = sorted(jobs, key=job_sort_key, reverse=True)

    scores = [clamp_int(j.get("fit_score"), 0) for j in jobs_sorted]
    avg_score = round(sum(scores) / len(scores), 1) if scores else 0

    gaps_counter: Counter = Counter()
    keywords_counter: Counter = Counter()
    rec_counter: Counter = Counter()

    for j in jobs_sorted:
        rec = str(j.get("final_recommendation") or "unknown")
        rec_counter[rec] += 1
        for g in norm_list(j.get("gaps")):
            gaps_counter[g] += 1
        for k in norm_list(j.get("missing_keywords")):
            keywords_counter[k] += 1

    now = datetime.now().strftime("%Y-%m-%d %H:%M")

    lines: List[str] = []
    lines.append(f"# Reporte ejecutivo - Matching de jobs\n")
    lines.append(f"Generado: {now}\n")

    lines.append("## Resumen ejecutivo\n")
    lines.append(f"- Jobs evaluados: **{len(jobs_sorted)}**")
    lines.append(f"- Score promedio: **{avg_score}**")
    if rec_counter:
        dist = ", ".join([f"{k}={v}" for k, v in rec_counter.most_common()])
        lines.append(f"- Distribución recomendación: **{dist}**")

    if jobs_sorted:
        best = jobs_sorted[0]
        lines.append("\n### Recomendación rápida\n")
        lines.append(f"- Top 1: `{best.get('job_url')}` (score={clamp_int(best.get('fit_score'), 0)}, rec={best.get('final_recommendation')})")

    lines.append("\n## Ranking (por score)\n")
    table_rows: List[List[str]] = [["#", "Job URL", "Score", "Senioridad", "Recomendación"]]
    for idx, j in enumerate(jobs_sorted, start=1):
        table_rows.append(
            [
                str(idx),
                md_escape(str(j.get("job_url") or "")),
                str(clamp_int(j.get("fit_score"), 0)),
                md_escape(str(j.get("seniority_guess") or "unknown")),
                md_escape(str(j.get("final_recommendation") or "unknown")),
            ]
        )
    lines.append(render_table(table_rows))

    lines.append("\n## Gaps y keywords más frecuentes\n")

    common_gaps = top_common(gaps_counter, top_common_n)
    common_kw = top_common(keywords_counter, top_common_n)

    if common_gaps:
        lines.append("### Gaps\n")
        for g, c in common_gaps:
            lines.append(f"- ({c}x) {g}")
    else:
        lines.append("### Gaps\n")
        lines.append("- (sin gaps repetidos reportados)")

    if common_kw:
        lines.append("\n### Missing keywords\n")
        for k, c in common_kw:
            lines.append(f"- ({c}x) {k}")
    else:
        lines.append("\n### Missing keywords\n")
        lines.append("- (sin missing keywords repetidas reportadas)")

    lines.append("\n## Acciones recomendadas (plan de 7 días)\n")
    lines.append("- Día 1: Ajustar CV a 1-2 versiones (Data/ML y Data Platform/Lakehouse).")
    lines.append("- Día 2: Optimizar LinkedIn: headline + about + skills (keywords de los roles target).")
    lines.append("- Día 3: Preparar 5 historias STAR (impacto, métricas, liderazgo, incidentes, arquitectura).")
    lines.append("- Día 4: Preparar portfolio rápido: 1-2 notebooks o repos con casos (churn, anomaly, streaming).")
    lines.append("- Día 5: Ensayo de entrevista técnica (ML + data engineering) y system design (lakehouse/streaming).")
    lines.append("- Día 6-7: Postular a los top jobs + networking con 5 contactos (mensaje corto + ajuste al rol).")

    lines.append("\n## Detalle por job\n")

    for j in jobs_sorted:
        url = str(j.get("job_url") or "")
        score = clamp_int(j.get("fit_score"), 0)
        seniority = str(j.get("seniority_guess") or "unknown")
        rec = str(j.get("final_recommendation") or "unknown")

        lines.append(f"### {url}\n")
        lines.append(f"- Score: **{score}**")
        lines.append(f"- Senioridad estimada: **{seniority}**")
        lines.append(f"- Recomendación: **{rec}**\n")

        pros = norm_list(j.get("pros"))
        gaps = norm_list(j.get("gaps"))
        missing = norm_list(j.get("missing_keywords"))
        bullets = norm_list(j.get("recommended_cv_bullets"))

        if pros:
            lines.append("**Fortalezas**")
            for p in pros[:8]:
                lines.append(f"- {p}")
            lines.append("")

        if gaps:
            lines.append("**Gaps / Riesgos**")
            for g in gaps[:10]:
                lines.append(f"- {g}")
            lines.append("")

        if missing:
            lines.append("**Keywords faltantes**")
            for k in missing[:15]:
                lines.append(f"- {k}")
            lines.append("")

        if bullets:
            lines.append("**Bullets sugeridos para el CV**")
            for b in bullets[:8]:
                lines.append(f"- {b}")
            lines.append("")

        actions = summarize_actions(j)
        if actions:
            lines.append("**Acciones para este job**")
            for a in actions:
                lines.append(f"- {a}")
            lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Genera un reporte ejecutivo (Markdown) a partir de job_match_report.json")
    parser.add_argument("--input", default="./outputs/job_match_report.json", help="Ruta al JSON de salida del matching")
    parser.add_argument("--output", default="./outputs/job_match_executive_report.md", help="Ruta del reporte Markdown")
    parser.add_argument("--top-common", type=int, default=10, help="Top N de gaps/keywords comunes")
    args = parser.parse_args()

    report = load_json(args.input)
    md = generate_markdown(report, top_common_n=int(args.top_common))

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        f.write(md)

    print(f"Reporte ejecutivo generado en: {args.output}")


if __name__ == "__main__":
    main()
