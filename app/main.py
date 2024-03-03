import os
from typing import Annotated
from fastapi import FastAPI, Path
from starlette.exceptions import HTTPException
from vdriver import VDriver

app = FastAPI()
@app.get("/clear_db")
def clear_content():
    result = VDriver().clear_db()
    if result is False:
        raise HTTPException(status_code=400, detail="Usually bad credentials for neo4j database or embeddings settings")
    return {"Content cleared"}

@app.get("/load_content_to_db")
def load_content():
    v = VDriver()
    content = ""
    try:
        for file in os.listdir('wiki'):
            if file.endswith(".txt"):
                with open(f'wiki/{file}') as f:
                    content += f.read()
    except FileNotFoundError:
        raise HTTPException(status_code=400, detail="folder wiki not found to process")
    results = v.vectorize(content=content)
    if results is False:
        raise HTTPException(status_code=400, detail="Usually bad credentials for neo4j database")
    return {"Content Loaded": results }

@app.get("/search/{query}/{k}")
def search(query: str, k:Annotated[int, Path(title="top K results", ge=1)]):
    v = VDriver()
    docs_with_score = v.query(query=query, size=k)
    if docs_with_score is not False:
        return [(document.page_content, score) for document, score in docs_with_score]

    raise HTTPException(status_code=400, detail="Usually bad credentials for neo4j database or embeddings settings")

