FROM pgvector/pgvector:pg16

# Install build dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    postgresql-server-dev-16 \
    libreadline-dev \
    zlib1g-dev \
    flex \
    bison \
    && rm -rf /var/lib/apt/lists/*

# Install Apache AGE
RUN git clone https://github.com/apache/age.git /tmp/age && \
    cd /tmp/age && \
    git checkout PG16 && \
    make USE_PGXS=1 && \
    make install && \
    rm -rf /tmp/age

# Copy initialization script
COPY init-extensions.sql /docker-entrypoint-initdb.d/

# Expose PostgreSQL port
EXPOSE 5432