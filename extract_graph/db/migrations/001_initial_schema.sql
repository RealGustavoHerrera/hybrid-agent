-- Create regular tables in public schema FIRST
CREATE TABLE IF NOT EXISTS documents (
    id SERIAL PRIMARY KEY,
    filename TEXT NOT NULL,
    content TEXT,
    processed BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP DEFAULT NOW(),
    processed_at TIMESTAMP DEFAULT NULL
);

-- NOW load AGE and create the graph
LOAD 'age';
SELECT ag_catalog.create_graph('knowledge_graph');