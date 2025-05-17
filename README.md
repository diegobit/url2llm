# url2llm

I needed a **super simple tool to crawl a website** (or the links in a *llms.txt*) into a formatted markdown file (without headers, navigation etc.) **to add to Claude or ChatGPT project documents**.

I haven't found an easy solution, there is some web based tool with a few free credits, but if you are already paying for some LLM with an api, why pay also someone else?

## Quickstart

#### With uv (recommended):

Thanks to uv, you can easily run it from anywhere without installing anything:

```bash
uvx url2llm \
   --depth 1 \
   --url "https://modelcontextprotocol.io/llms.txt" \
   --instruction "I need documents related to developing MCP (model context protocol) servers" \
   --provider "gemini/gemini-2.5-flash-preview-04-17" \
   --api_key ${GEMINI_API_KEY}
```

Then drag `./model-context-protocol-documentation.md` into ChatGPT/Claude!

> [!TIP]
> You can invoke it with `url2llm` as a properly installed cli tool after running `uv tool install url2llm`.

#### With pip (alternative):

```
pip install url2llm
```

## What it does

The script uses Crawl4AI:

1. For each url in the crawling, the script produces a markdown
2. Then it asks the LLM to extract from each page only the content relevant to the given instruction.
3. Merge all pages into one and save the merged file.

## Command args and hints

- To use **another LLM provider**, just change `--provider` to eg. `openai/gpt-4o`
   - always set `--api-key`, it is not always inferred correctly fron env vars
- Provide a **clear goal** to `--instruction`. This will guide the LLM to filter out irrelevant pages.
- Recommended **depth** (default = `2`):
   - `2` or `1` for normal website
   - `1` for llms.txt
- Provide `--output_dir` to change where files are saved (default = `.`)
- If you need the single pages, use `--keep_pages True` (default = `False`)
- You can specify the **concurrency** with `--concurrency` (default = `16`)
- The scripts deletes files **shorter** than `--min_chars` (default = `1000`)

> [!CAUTION]
> If you need to do more complex stuff use Crawl4AI directly and build it yourself: https://docs.crawl4ai.com/
