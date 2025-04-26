# website2llm

I needed a **super simple tool to crawl a website** (or the links in a *llms.txt*) into formatted markdown files (without headers, navigation etc.) **to add to Claude or ChatGPT project documents**.

I haven't found an easy solution, there is some web based tool with a few free credits, but if you are already paying for some LLM with an api, why pay also someone else?

## What it does

The script uses Crawl4AI:

1. For each url in the crawling, the script produces a markdown
2. Then it asks the LLM to extract only the content relevant to the given instruction and save all files to disk.
3. Merge all files into one and save the merged file.

## Installation

1. Clone the repo, then:

   - **(Recommended, with uv)** – Nothing to do

   - **(Alternative, with pip)** – Install `crawl4ai` and `fire`

## How to use

### Run script with arguments:

```bash
uv run main.py \
   --url "<URL_OR_LLMS.TXT>" \
   --depth 1 \
   --instruction "I need documents related to <GOAL>" \
   --provider "<PROVIDER>/<MODELNAME>" \
   --api-key ${GEMINI_API_KEY} \
   --output-dir "<OUTPUT_DIR>"
```

- To use **another LLM provider**, just change `--provider` to eg. `openai/gpt-4o`
   - always set `--api-key`, it is not always inferred correctly fron env vars
- Provide a **clear goal** to `--instruction`. This will guide the LLM to filter out irrelevant pages.
- Recommended **depth** (default = `2`):
   - `2` or `1` for normal website
   - `1` for llms.txt
- You can specify the **concurrency** with `--concurrency` (default = 16)
- The scripts deletes files **shorter** than `--min_chars` (Default = 1000)

> [!CAUTION]
> If you need to do more complex stuff use Crawl4AI directly and build it yourself: https://docs.crawl4ai.com/

### How I use it

Thanks to uv, I can easily run it from anywhere in my system:

```bash
uv \
   --directory ~/Dev/website2llm/ \
   run main.py \
   --url "https://modelcontextprotocol.io/llms.txt" \
   --instruction "I need documents related to developing MCP (model context protocol) servers" \
   --provider "gemini/gemini-2.5-flash-preview-04-17" \
   --api_key ${GEMINI_API_KEY} \
   --output-dir ~/Desktop/crawl_out/
```

And drag `~/Desktop/crawl_out/merged/model-context-protocol-documentation.md` into ChatGPT/Claude!
