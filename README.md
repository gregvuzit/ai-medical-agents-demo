# AI Medical Agents Example

This application for MacOS demonstrates OpenAI's Swarm technique: <https://github.com/openai/swarm>

It takes a user prompt describing symptoms of an illness and will return a best guess diagnosis and then a prescription based on said diagnosis, utilizing a RAG store containing embedded text from the two included PDF documents.

Based off of code in <https://github.com/pdichone/ollama-fundamentals> (specifically `pdf-rag-streamlit.py`) and <https://github.com/pdichone/swarm-writer-agents>.

## Prerequisites

Latest version of MacOS

Python: <https://www.python.org/downloads/>

Ollama: <https://ollama.com/>

Make sure you have pulled at least one model into your local Ollama installation.

Install the **poppler**, **libheif**, and **tesseract** packages from homebrew:
`brew install poppler`
`brew install libheif`
`brew install tesseract`

## Setup

1. Clone the repo and go to the root directory in a terminal window.
2. Create virtual enviroment: `python3 -m venv venv`
3. `source venv/bin/activate` (`decactivate` to turn off)
4. Install python packages: `pip install -r requirements.txt`
5. Start app: `python -m streamlit run app.py`


Make sure the `unstructured` package is version `0.16.12` and `nltk` is `3.9.1` or you will run into this: <https://github.com/Unstructured-IO/unstructured/issues/3511> (see <https://github.com/Unstructured-IO/unstructured/issues/3511#issuecomment-2603714045>)

On first load, the app will setup the RAG store in the `chroma_db` directory off the root. This will take several minutes.