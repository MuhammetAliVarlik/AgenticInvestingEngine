#!/bin/bash

export OLLAMA_MODELS=/ollama

# Start Ollama
echo "🚀 Starting Ollama server..."
/bin/ollama serve &

pid=$!

# Is Ollama ready? Check using `ollama list`
echo "⏳ Waiting for Ollama to be ready..."
until ollama list >/dev/null 2>&1; do
  sleep 1
done
echo "✅ Ollama is ready."

# Is the model installed?
if ! ollama list | grep -q "llama3.1"; then
  echo "🔽 Pulling llama3.1:latest model..."
  ollama pull llama3.1:latest
else
  echo "📦 Model llama3.1 already exists. Skipping download."
fi

# Run the model
echo "▶️ Running llama3.1 model..."
ollama run llama3.1:latest --keepalive -1m
echo "✅ Model is running."

# Track the lifecycle of the Ollama process
wait $pid
