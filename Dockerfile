# ponytail: single stage; pip install from source is the whole build.
# Split into a builder stage if image size ever matters.
FROM python:3.12-slim

WORKDIR /src
COPY pyproject.toml LICENSE README.md ./
COPY forge/ forge/
COPY evals/ evals/
# evals/ must sit next to the installed package (forge/evals.py resolves
# EVALS_DIR relative to forge/__file__) so `forge eval` works in-container
RUN pip install --no-cache-dir . \
    && cp -r evals "$(python -c 'import forge, pathlib; print(pathlib.Path(forge.__file__).resolve().parent.parent)')/evals" \
    && rm -rf /src

ENV FORGE_DB=/data/forge.db \
    FORGE_BIND=0.0.0.0
VOLUME /data
EXPOSE 8765
CMD ["forge", "web", "--port", "8765"]
