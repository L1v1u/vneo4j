# Vneo4j

Vneo4j is a Python app that use fastapi , langchain, neo4j.

the app vectorize a set of texts and create indexed vector/graph DB from it.  

## Installation

then first time

```code
docker-compose up --build
```

## Usage

on [http://localhost:8083/docs#/](http://localhost:8083/docs#/)

1. execute [GET] /load_content_to_db 

   first to vectorise the wiki folder content 

2. to query use 

   [GET] /search/{query}/{k}

## License

[MIT](https://choosealicense.com/licenses/mit/)