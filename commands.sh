mitmweb -p 11435 --mode reverse:http://localhost:11434 -w ollama_traffic.bin
ollama run deepseek-r1:14b
ollama run qwen2.5:7b