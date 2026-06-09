-- HireLens PostgreSQL initialization
-- Creates the MLflow tracking database alongside the main app database

CREATE DATABASE hirelens_mlflow
    WITH
    OWNER = hirelens_user
    ENCODING = 'UTF8'
    LC_COLLATE = 'en_US.utf8'
    LC_CTYPE = 'en_US.utf8'
    TEMPLATE = template0;

GRANT ALL PRIVILEGES ON DATABASE hirelens TO hirelens_user;
GRANT ALL PRIVILEGES ON DATABASE hirelens_mlflow TO hirelens_user;

-- Enable pg_trgm for fuzzy text search on resume/job fields
\c hirelens
CREATE EXTENSION IF NOT EXISTS pg_trgm;
CREATE EXTENSION IF NOT EXISTS btree_gin;
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
