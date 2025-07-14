FROM python:3.12-slim

# Install system dependencies for building packages
RUN apt-get update && apt-get install -y curl build-essential

# Install Poetry
RUN curl -sSL https://install.python-poetry.org | python3 -
ENV PATH="/root/.local/bin:$PATH"

# Set working directory
WORKDIR /app

# Copy only pyproject.toml and poetry.lock first (to leverage Docker cache)
COPY pyproject.toml poetry.lock* /app/

# Install dependencies
RUN poetry config virtualenvs.create false \
    && poetry install --no-interaction --no-ansi --no-root

# Copy the rest of the code
COPY ./src /app/src
COPY ./config /app/config

# Expose port
EXPOSE 80
EXPOSE 5000
EXPOSE 8000
EXPOSE 4200

