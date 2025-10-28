FROM python:3.10-slim

WORKDIR /app

RUN pip install --no-cache-dir numpy matplotlib ase mace-torch

COPY mace_ase_relax-docker.py .
COPY models/mace-omat-0-small-fp32.model /app/models/mace-omat-0-small-fp32.model

ENV MACE_MODEL_PATH=/app/models/mace-omat-0-small-fp32.model

ENTRYPOINT ["python", "./mace_ase_relax-docker.py"]
