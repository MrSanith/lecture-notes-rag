# LectureLens — Chat with Lecture Notes

A local Retrieval-Augmented Generation application that answers
questions from uploaded lecture PDFs and displays supporting passages
with filenames and PDF page references.

## Features

- Upload multiple text-based PDFs
- Extract text with PDF page metadata
- Token-based chunking with overlap
- Semantic retrieval using normalized embeddings
- Local answer generation through Ollama
- Source identifiers and inspectable evidence
- Insufficient-evidence response
- Downloadable conversation history

## Architecture

PDF upload → page extraction → token chunks → embeddings →
cosine similarity retrieval → local LLM → answer and evidence

## Technology

Python, Streamlit, pypdf, Sentence Transformers, NumPy,
Requests, Ollama, and Llama 3.2.

## Setup on Windows

Use Python 3.11 as the project's initial setup target.

```powershell
git clone https://github.com/YOUR_USERNAME/lecture-notes-rag.git
cd lecture-notes-rag
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Install Ollama separately, then download the model:

```powershell
ollama pull llama3.2:3b
```

Start Ollama and launch the application:

```powershell
.\.venv\Scripts\python.exe -m streamlit run app.py
```

The embedding model downloads on first use.
Local performance depends on available hardware.

## Usage

1. Upload text-based PDFs.
2. Build the knowledge base.
3. Ask a complete question.
4. Inspect the answer and retrieved passages.
5. Rebuild after changing the selected documents.

## Tests

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

## Evaluation

See docs/evaluation.md for questions, settings, results, and failures.

Add actual results here after evaluation.

## Limitations

- Scanned PDFs require OCR, which is not included.
- Tables, equations, and complex layouts may extract poorly.
- Chat history is displayed but is not used to resolve follow-up questions.
- The search index is held in memory and rebuilt after restarting.
- Similarity thresholds require evaluation.
- Valid source identifiers do not prove factual support.
- The language model can still generate unsupported statements.
- Intended as a local, single-user portfolio prototype.

## Data

Uploaded PDFs, generated indexes, and local configuration are excluded
from version control. Use documents you have permission to process.
Public demo materials should be self-authored or openly licensed.

## License

Application code is available under the MIT License.
Third-party models and dependencies retain their respective licenses.
