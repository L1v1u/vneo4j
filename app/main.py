import os
from typing import Annotated
from fastapi import FastAPI, Path
from vdriver import VDriver

app = FastAPI()
@app.get("/clear_db")
def clear_content():
    v = VDriver().clear_db()


@app.get("/load_content_to_db")
def load_content():
    v = VDriver()
    content = ""
    for file in os.listdir('wiki'):
        if file.endswith(".txt"):
            with open(f'wiki/{file}') as f:
                content += f.read()

    results = v.vectorize(content=content)
    return {"Content Loaded": results }


@app.get("/search/{query}/{k}")
def search(query: str, k:Annotated[int, Path(title="top K results", ge=1)]):
    v = VDriver()
    docs_with_score = v.query(query=query, size=k)
    return [(document.page_content, score) for document, score in docs_with_score]
