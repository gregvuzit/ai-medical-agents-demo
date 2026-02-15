# AI Medical Agents Example

This application for MacOS demonstrates OpenAI's Swarm technique: <https://github.com/openai/swarm>

It takes a user prompt describing symptoms of an illness and will return a best guess diagnosis and then a prescription based on said diagnosis, utilizing a RAG store containing embedded text from the two included PDF documents.

Based off of code in <https://github.com/pdichone/ollama-fundamentals> (specifically `pdf-rag-streamlit.py`) and <https://github.com/pdichone/swarm-writer-agents>.

## Prerequisites

MacOS 15 or above

Python 3.13: <https://www.python.org/downloads/>

Ollama: <https://ollama.com/>

Make sure you have pulled at least one model into your local Ollama installation.

Install the **poppler**, **libheif**, and **tesseract** packages from homebrew:

`brew install poppler`

`brew install libheif`

`brew install tesseract`

Run the following command to ensure that your Python installation has the proper CA certificates and links (needed to download the tokenizer data file below):
`/Applications/Python\ 3.13/Install\ Certificates.command`

Make sure Ollama is running on the machine.

## Setup

1. Clone the repo and go to the root directory in a terminal window.
2. Create virtual enviroment: `python3 -m venv venv`
3. `source venv/bin/activate` (`decactivate` to turn off)
4. Install python packages: `python3 -m pip install -r requirements.txt`
5. Open up a python terminal: `python 3`

Run the following and exit():

```
import nltk

nltk.download("punkt")
```

6. Start app: `python3 -m streamlit run app.pyy`

On first load, the app will setup the RAG store in the `chroma_db` directory off the root. This will take anywhere from around 15-30 minutes.