# syntax=docker/dockerfile:1
#
# Python plotting image for benchmark visualization (matplotlib). Kept separate
# from the Rust images so the toolchains don't mix.

FROM python:3.12-slim
RUN pip install --no-cache-dir matplotlib==3.9.2
WORKDIR /work
ENTRYPOINT ["python", "benchmarks/plot.py"]
