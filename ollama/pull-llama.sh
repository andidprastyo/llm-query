# ollama/pull-llama.sh
#!/bin/bash
set -e

# Start Ollama server
ollama serve &

# Wait for Ollama server to start
sleep 5

# Pull the model
echo "Pulling llama3.2:1b model..."
ollama pull llama3.2:1b

# Keep container running
wait