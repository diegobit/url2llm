# website2llm

**Why?**

I needed a **one line tool to crawl a website** (or a single webpage) into a formatted markdown, without headers, navigation etc. **to add to Claude or ChatGPT project documents**.

I haven't found an easy solution, there is some web based tool with a few free credits, but if you are already paying for some LLM with an api, why pay also someone else?

## What it does

This is a thin wrapper around Crawl4AI that:

1. crawls the given url and goes deeper by the provided `--depth` (default = 1). 0 means single webpage
2. For each url, Crawl4AI produces a markdown, then the script asks the LLM to only extract relevant content, according to `--instruction`. Unlike Crawl4AI, it does it afterwards.
3. Keeps only files longer than `--md_min_chars` (default = 1000)

## Installation

**Recommended, with uv:** Nothing to do
**Alternative, pip:** install `crawl4ai` and `fire`

## How to use

### Example run

```bash
uv run main.py \
   --url "https://modelcontextprotocol.io/docs/" \
   --depth 1 \
   --instruction "I need documents related to developing MCP (model context protocol) servers" \
   --provider "gemini/gemini-2.5-flash-preview-04-17" \
   --api-key ${GEMINI_API_KEY} \
   --concurrency 32 \
   --output-dir ./md_out
```

- To use another LLM provider, just change `--provider` to eg. `openai/gpt-4o` (also set --llm-api-key)
- Provide a clear goal to `--instruction`. This will guide the LLM to filter out irrelevant pages.

> [!CAUTION]
> If you need to do more complex stuff use Crawl4AI directly and build it yourself: https://docs.crawl4ai.com/

