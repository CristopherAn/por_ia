# RAG Matching de Empleos con Ollama

Este proyecto compara un perfil profesional en texto contra varias descripciones de empleo en `.txt` usando embeddings con Ollama, almacenamiento en ChromaDB y evaluacion final con un LLM local.

El flujo actual del repo es local y basado en archivos. No usa scraping de URLs ni scripts de LinkedIn.

## Flujo real

1. `index_jobs.py` carga `inputs/job_*.txt`, los divide en chunks y los guarda en ChromaDB.
2. `match_jobs.py` carga `inputs/profile.txt`, recupera desde Chroma los fragmentos mas relevantes de cada job y genera una evaluacion final en JSON.
3. `generate_cv.py` crea un CV nuevo en Markdown considerando todos los jobs y el reporte de matching.
4. Las salidas quedan en `outputs/job_match_report.json`, `outputs/tailored_cv.json` y `outputs/tailored_cv.md`.

## Archivos clave

- `config/rag_config.json`: configuracion principal
- `index_jobs.py`: indexacion de jobs en ChromaDB
- `match_jobs.py`: matching perfil vs jobs
- `generate_cv.py`: generacion de CV consolidado
- `rag_utils.py`: utilidades de embeddings, Chroma y chunking
- `inputs/profile.txt`: perfil del candidato
- `inputs/job_*.txt`: descripciones de trabajo

## Requisitos

- Python 3.10+
- Ollama corriendo localmente
- Modelos disponibles en Ollama:
  - chat: `llama3.2:3b-instruct-fp16`
  - embeddings: `nomic-embed-text`

Instalacion:

```bash
pip install -r requirements.txt
```

## Configuracion

Archivo: `config/rag_config.json`

```json
{
  "rag": {
    "provider": "ollama",
    "persist_directory": "./chroma_url_db",
    "collection_name": "url_docs",
    "artifacts_dir": "./rag_artifacts",
    "ollama_base_url": "http://localhost:11434",
    "ollama_chat_model": "llama3.2:3b-instruct-fp16",
    "ollama_embedding_model": "nomic-embed-text",
    "chunk_size": 1000,
    "chunk_overlap": 200,
    "top_k": 4,
    "reset_db": false
  },
  "match": {
    "profile_text_path": "./inputs/profile.txt",
    "job_text_glob": "./inputs/job_*.txt",
    "output_path": "./outputs/job_match_report.json",
    "prompt_template": "..."
  },
  "cv": {
    "output_json_path": "./outputs/tailored_cv.json",
    "output_markdown_path": "./outputs/tailored_cv.md"
  }
}
```

Variables de entorno utiles:

- `RAG_PROVIDER`
- `RAG_PERSIST_DIRECTORY`
- `RAG_ARTIFACTS_DIR`
- `RAG_OLLAMA_BASE_URL`
- `RAG_OLLAMA_CHAT_MODEL`
- `RAG_OLLAMA_EMBEDDING_MODEL`
- `RAG_CHUNK_SIZE`
- `RAG_CHUNK_OVERLAP`
- `RAG_TOP_K`

## Uso

Indexar jobs:

```bash
python index_jobs.py --config config/rag_config.json
```

Cada corrida reemplaza la base anterior en `chroma_url_db`.

Generar reporte:

```bash
python match_jobs.py --config config/rag_config.json
```

`match_jobs.py` espera que la base ya exista. Si no indexaste antes, primero corre `index_jobs.py`.

Generar CV consolidado:

```bash
python generate_cv.py --config config/rag_config.json
```

## Salida

Archivo: `outputs/job_match_report.json`

Ejemplo de resultado por job:

```json
{
  "job_source": "./inputs/job_1.txt",
  "fit_score": 78,
  "seniority_guess": "senior",
  "pros": ["Experiencia fuerte en Python"],
  "gaps": ["Falta experiencia en Kubernetes"],
  "missing_keywords": ["Docker", "CI/CD"],
  "recommended_cv_bullets": ["Agregar proyecto con despliegues en produccion"],
  "final_recommendation": "apply"
}
```

Archivos de CV:

- `outputs/tailored_cv.json`: resumen estructurado de requisitos y cobertura
- `outputs/tailored_cv.md`: CV final en Markdown

## Docker

El contenedor esta preparado para usar Ollama del host en Windows mediante `http://host.docker.internal:11434`.

Construir:

```bash
docker compose build
```

Indexar:

```bash
docker compose run --rm rag python index_jobs.py --config config/rag_config.json
```

Generar reporte:

```bash
docker compose run --rm rag python match_jobs.py --config config/rag_config.json
```

Generar CV:

```bash
docker compose run --rm rag python generate_cv.py --config config/rag_config.json
```

## Troubleshooting

### Error: `Failed to connect to Ollama`

En host local usa `http://localhost:11434`.

En Docker usa `http://host.docker.internal:11434`.

### Error: `No existe la base vectorial`

Primero ejecuta:

```bash
python index_jobs.py --config config/rag_config.json
```

### Error: salida JSON invalida del modelo

`match_jobs.py` intenta extraer el JSON y, si el modelo responde con formato defectuoso, hace un segundo intento de reparacion. Aun asi, conviene mantener `prompt_template` bien estricto.
