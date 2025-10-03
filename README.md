# Extract and Graph

Receive text files  (ie. a transcript of a conversation or interview) 
Vectorize them
Extract structured information from unstructured text and create a graph with the information
Use the vectors and the graph to create a Hybrid RAG Agent


# Installing
In the same folder than the pyproject.toml

```bash
$ (venv) python -m pip install -e .
$ (venv) python -m pip install apache-age-python --no-deps
```

The reason for `apache-age-python` to be excluded is that it tries to compile psycopg2 from source and it fails for reasons.
Installing it via pip with `--no-deps` solves the issue (but unfortunatelly there is no way to declare a `--no-deps` in the `pyproject.toml` file).

# Installing with dependencies
With the optional `".[dev]"` parameter will install dev dependencies as well

```bash
$ (venv) python -m pip install -e ".[dev]"
$ (venv) python -m pip install apache-age-python --no-deps
```

# Starting the Database

docker-compose.yaml will use Dockerfile to create an image with pgvector and AGE installed.
Then it will mount the database in the local ./database folder
If it's the first time it's better to `build` first and then `up` the container
```
docker-compose build --no-cache
docker-compose up -d
```
