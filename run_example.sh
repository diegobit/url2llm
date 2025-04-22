uv run main.py \
   --url "https://modelcontextprotocol.io/" \
   --depth 2 \
   --instruction "I need documents related to developing MCP (model context protocol) servers" \
   --provider "gemini/gemini-2.5-flash-preview-04-17" \
   --api_key ${GEMINI_API_KEY} \
   --concurrency 32 \
   --output-dir ~/Desktop/crawl_out/
