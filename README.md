# RAG Matching de Empleos - Ollama

Este proyecto procesa archivos locales (.txt) de perfiles y descripciones de trabajos para hacer matching usando **RAG** (retrieval + generación) con **Chroma** y **Ollama**.

Soporta el proveedor:
- `ollama` (local, por defecto `http://localhost:11434`)

## Arquitectura (Indexación → Matching)

```mermaid
flowchart TD
    A[Usuario<br/>Inicia el proceso] --> B[Configuración<br/>Carga settings desde rag_config.json<br/>Archivo: config/rag_config.json]
    B --> C[Inputs<br/>Archivos de Jobs desde ./inputs/<br/>Jobs: job_*.txt<br/>Profile: profile.txt<br/>Archivos: inputs/]
    
    C --> D[Indexing<br/>1. Cargar jobs desde archivos .txt<br/>2. Split en chunks<br/>3. Generar embeddings con Ollama<br/>4. Almacenar en ChromaDB<br/>Código: index_jobs.py<br/>Restablece DB cada indexación]
    
    D --> E[Matching<br/>1. Cargar perfil<br/>2. Retrieve chunks relevantes<br/>3. Evaluar con LLM<br/>4. Generar reporte JSON<br/>Código: match_jobs.py]
    
    G[Proveedor IA<br/>Ollama<br/>Embedding: nomic-embed-text<br/>Chat: llama3.2:3b<br/>http://localhost:11434]
    
    G --> D
    G --> E
    
    E --> H[Output<br/>job_match_report.json<br/>Archivo: outputs/]
```

## Archivos Clave

- **`config/rag_config.json`**: Configuración de RAG + prompt para evaluación de jobs
- **`index_jobs.py`**: Indexa jobs en ChromaDB (entrena la base vectorial)
- **`match_jobs.py`**: Evalúa perfil vs jobs indexados + genera reporte
- **`rag_utils.py`**: Utilidades core: embeddings, split, indexing, LLM

## Flujo de Uso

```bash
# 1. Indexación (entrena la BD - ELIMINA la anterior)
python index_jobs.py --config config/rag_config.json

# 2. Matching (evalúa perfil vs jobs + genera reporte)
python match_jobs.py --config config/rag_config.json
```

**Nota**: Cada ejecución de `index_jobs.py` borra automáticamente la BD anterior (no requiere flag).

## Requisitos

- Python 3.10+
- Ollama instalado y ejecutándose localmente
- Archivos de entrada (.txt) en `inputs/`:
  - `profile.txt` - Perfil del candidato
  - `job_*.txt` - Descripciones de trabajos (matching con glob)

## Instalación

```bash
pip install -r requirements.txt
```

## Configuración (`config/rag_config.json`)

```json
{
  "rag": {
    "provider": "ollama",
    "ollama_base_url": "http://localhost:11434",
    "ollama_embedding_model": "nomic-embed-text",
    "ollama_chat_model": "llama3.2:3b-instruct-fp16",
    "persist_directory": "./chroma_url_db",
    "chunk_size": 1000,
    "chunk_overlap": 200
  },
  "match": {
    "profile_text_path": "./inputs/profile.txt",
    "job_text_glob": "./inputs/job_*.txt",
    "output_path": "./outputs/job_match_report.json",
    "prompt_template": "..."
  }
}
```

## Output (`job_match_report.json`)

Ejemplo de resultado para cada job evaluado:
```json
{
  "job_source": "./inputs/job_1.txt",
  "fit_score": 78,
  "seniority_guess": "senior",
  "pros": ["Strong Python experience", "Leadership skills"],
  "gaps": ["Missing Kubernetes", "No microservices"],
  "missing_keywords": ["Docker", "CI/CD"],
  "recommended_cv_bullets": ["Add Kubernetes project details"],
  "final_recommendation": "apply"
}
```


- `langchain`
- `langchain-community`
- `langchain-text-splitters`
- `langchain-chroma`
- `chromadb`
- `langchain-ollama`

Alternativa: podés usar Docker/Compose (ver sección "Docker").

## Configuración

El archivo `./config/rag_config.json` contiene defaults del proyecto. Podés cambiarlos allí o sobreescribir por CLI (flag `--config`).

Variables de entorno útiles:

- `RAG_PROVIDER`
- `RAG_PERSIST_DIRECTORY`
- `RAG_ARTIFACTS_DIR`
- `RAG_CHUNK_SIZE`
- `RAG_CHUNK_OVERLAP`
- `RAG_TOP_K`

## Flujo por etapas

El pipeline tiene 3 etapas:

1) `fetch`: descarga + limpia y guarda `documents.json`
2) `chunk`: genera `chunks.json`
3) `index`: genera embeddings y persiste en Chroma

Los artefactos quedan en `./rag_artifacts/url/`.

## Matching: perfil vs lista de jobs (LinkedIn)

Este repo incluye `match_linkedin_jobs.py`, que toma:

- un link de **perfil** (`match.profile_url`)
- una **lista de links** de jobs (`match.job_urls`)

y genera un reporte con compatibilidad (score + razones) en `match.output_path`.

Configuración (ver `./config/rag_config.json`):

- `match.profile_url`
- `match.job_urls` (lista)
- `match.output_path`
- `match.prompt_template` (el prompt que fuerza salida JSON)

Ejecutar en host:

```bash
python match_linkedin_jobs.py
```

Forzar re-descarga (ignorar cache en `rag_artifacts`):

```bash
python match_linkedin_jobs.py --refresh
```

### Ejemplo con Ollama (100% local)

Asegurate de tener modelos:

- Chat: `llama3.2:3b-instruct-fp16`
- Embeddings: `nomic-embed-text`

Etapas:

```bash
python "Guárdalo como rag_url_langchain.py" --provider ollama fetch --url "https://TU_URL"
python "Guárdalo como rag_url_langchain.py" --provider ollama chunk
python "Guárdalo como rag_url_langchain.py" --provider ollama --reset-db index --url "https://TU_URL"
python "Guárdalo como rag_url_langchain.py" --provider ollama ask --question "¿De qué trata la página?"
```

Modo chat:

```bash
python "Guárdalo como rag_url_langchain.py" --provider ollama chat
```

### Ejemplo con OpenAI

Configurar API key:

- PowerShell:

```powershell
$env:OPENAI_API_KEY="tu_api_key"
```

Ejecutar:

```bash
python "Guárdalo como rag_url_langchain.py" --provider openai index --url "https://TU_URL"
python "Guárdalo como rag_url_langchain.py" --provider openai ask --question "..."
```

## Notas

- Algunas páginas (LinkedIn, sitios con login/JS pesado) pueden no cargar bien con `WebBaseLoader`.
- Si re-indexás la misma URL varias veces sin resetear, podés introducir duplicados. Usá `--reset-db` cuando quieras recrear la base.

## Docker

Incluye:

- `Dockerfile`
- `docker-compose.yml`

El `docker-compose.yml` está configurado para usar Ollama del host en Windows vía:

- `http://host.docker.internal:11434`

Importante: si tu `rag_config.json` tiene `ollama_base_url` en `http://localhost:11434`, dentro del contenedor eso apunta al propio contenedor (y va a fallar). En Docker usá `host.docker.internal`.

Nota: por defecto los scripts leen `./config/rag_config.json`. Podés sobreescribir la ruta con `--config`.

### Ejecutar el pipeline en contenedor (runner)

Construir imagen:

```bash
docker compose build
```

Ejecutar pipeline completo (fetch -> chunk -> index) con pregunta final:

```bash
docker compose run --rm rag python run_rag_pipeline.py --ollama-base-url "http://host.docker.internal:11434" --url "https://TU_URL" --question "¿De qué trata la página?"
```

Ejecutar modo chat (después de indexar):

```bash
docker compose run --rm rag python run_rag_pipeline.py --ollama-base-url "http://host.docker.internal:11434" --url "https://TU_URL" --chat
```

### Ejecutar el CLI en contenedor

También podés ejecutar el CLI por etapas:

```bash
docker compose run --rm rag python "Guárdalo como rag_url_langchain.py" --ollama-base-url "http://host.docker.internal:11434" fetch --url "https://TU_URL"
docker compose run --rm rag python "Guárdalo como rag_url_langchain.py" chunk
docker compose run --rm rag python "Guárdalo como rag_url_langchain.py" --ollama-base-url "http://host.docker.internal:11434" --reset-db index --url "https://TU_URL"
docker compose run --rm rag python "Guárdalo como rag_url_langchain.py" ask --question "..."
```

### Ejecutar matching en contenedor

```bash
docker compose run --rm rag python match_linkedin_jobs.py
```

Con refresh:

```bash
docker compose run --rm rag python match_linkedin_jobs.py --refresh
```

## Troubleshooting

### Error: `Failed to connect to Ollama`

Si estás en Docker, no uses `localhost` para Ollama.

- Usá `--ollama-base-url "http://host.docker.internal:11434"` en el comando, o
- Editá `rag_config.json` y cambiá `ollama_base_url` a `http://host.docker.internal:11434`.

En host (sin Docker) sí corresponde `http://localhost:11434`.

### Error: `No module named 'bs4'`

`WebBaseLoader` necesita BeautifulSoup. Este repo incluye `beautifulsoup4` y `lxml` en `requirements.txt`. Si instalaste dependencias manualmente, reinstalá con:

```bash
pip install -r requirements.txt
```

### LinkedIn

LinkedIn puede devolver contenido incompleto o redirigir a login por anti-bot/JS.
Si no hay texto útil, la alternativa más estable es indexar desde:

- texto copiado de la descripción del job, o
- HTML guardado desde el navegador.

### Warnings de telemetry / onnxruntime

- Mensajes de telemetry de Chroma pueden aparecer como warnings y no siempre bloquean.
- Warnings de `onnxruntime` sobre GPU en Docker se pueden ignorar si el proceso continúa.
