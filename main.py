"""
Crawl a website and export topic‑focused markdown files (one per page) using Crawl4AI.
"""

import asyncio
import functools
import re
from pathlib import Path
from typing import List, Optional

import aiohttp
import fire
from crawl4ai import (
    AsyncWebCrawler,
    BrowserConfig,
    CacheMode,
    CrawlerRunConfig,
    LLMConfig,
)
from crawl4ai.content_filter_strategy import LLMContentFilter
from crawl4ai.deep_crawling import BFSDeepCrawlStrategy
from crawl4ai.markdown_generation_strategy import DefaultMarkdownGenerator


_slug_rx = re.compile(r"[^\w-]+", re.UNICODE)


def slugify(url: str) -> str:
    url = re.sub(r"^https?://", "", url).rstrip("/")
    slug = _slug_rx.sub("-", url)
    return slug[:120].strip("-") or "page"

def write_markdown(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")

async def filter_content_async(llm_filter, raw_md):
    """Run the blocking .filter_content in a thread‑pool executor."""
    loop = asyncio.get_running_loop()
    func = functools.partial(llm_filter.filter_content, raw_md)
    return await loop.run_in_executor(None, func)

async def process_page(
    res,
    llm_filter,
    output_dir: Path,
    semaphore: asyncio.Semaphore,
    min_chars: int
) -> None:
    async with semaphore:
        # Skip failure or no content
        if not res.success or not getattr(res, "markdown", None):
            return

        raw_md = (
            getattr(res.markdown, "raw_markdown", None)
            or (res.markdown if isinstance(res.markdown, str) else None)
        )
        if not raw_md:
            return

        # Clean with LLM
        md_text = await filter_content_async(llm_filter, raw_md)
        md_text = "\n\n".join(md_text)

        # Skip too-short outputs
        if not md_text or len(md_text.strip()) < min_chars:
            return

        filename = f"{slugify(res.url)}.md"
        write_markdown(output_dir / filename, md_text)
        print(f"Saved → {output_dir/filename}")

async def parse_llms_txt(url: str) -> List[str]:
    """Fetch and parse an llms.txt file, returning a list of URLs."""
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            if response.status != 200:
                print(f"Failed to fetch llms.txt: {response.status}")
                return []

            content = await response.text()
            url_pattern = re.compile(r'\]\((https?://[^)]+)\)')
            urls = url_pattern.findall(content)

            print(f"Found {len(urls)} URLs in llms.txt")
            return urls

async def crawl_and_export(
    base_url: str,
    instruction: str,
    output_dir: Path,
    depth: int,
    model: str,
    concurrency: int,
    api_key: Optional[str],
    md_min_chars: int
) -> None:
    """Crawl *base_url* and write cleaned, relevant markdown files under *output_dir*."""

    llm_cfg = LLMConfig(provider=model, api_token=api_key)

    # LLM filter: extract relevant content and skip unrelated pages
    llm_filter = LLMContentFilter(
        llm_config=llm_cfg,
        instruction=f"""
{instruction}

# Important rules:
- If the page contains no information relevant to the instruction above, return an empty string (skip page).
- If the page is relevant to the instruction:
    - Omit navigation menus, sidebars, footers, cookie banners, ads.
    - Keep headings, lists, code blocks, and tables intact.
    - Ensure each heading (e.g., '#', '##') appears on its own line, followed by a blank line.
    - Preserve original paragraph breaks and blank lines between sections.
    - Return clean, well-structured markdown only.
""",
        chunk_token_threshold=8192,
        verbose=True,
    )

    # Use plain Markdown generator (raw conversion) and apply LLM filter manually
    md_generator = DefaultMarkdownGenerator(
        options={"ignore_links": True},
    )

    deep_strategy = BFSDeepCrawlStrategy(max_depth=depth, include_external=False)

    run_cfg = CrawlerRunConfig(
        deep_crawl_strategy=deep_strategy,
        markdown_generator=md_generator,
        cache_mode=CacheMode.BYPASS,
        semaphore_count=concurrency,
        verbose=True,
    )

    start_urls = []
    if base_url.lower().endswith('llms.txt'):
        print(f"Processing llms.txt from {base_url}")
        start_urls = await parse_llms_txt(base_url)
        if not start_urls:
            print(f"No valid URLs found in {base_url}")
            return
    else:
        start_urls = [base_url]

    results = []
    async with AsyncWebCrawler(config=BrowserConfig(headless=True)) as crawler:
        for url in start_urls:
            print(f"Crawling {url} (depth={depth})...")
            results_i = await crawler.arun(url, config=run_cfg)
            results.extend(results_i)
            print(f"Found {len(results)} pages from {url}")

        # results: List = await crawler.arun(base_url, config=run_cfg)

    # Create semaphore for LLM processing
    semaphore = asyncio.Semaphore(concurrency)
    tasks = [
        process_page(res, llm_filter, output_dir, semaphore, md_min_chars)
        for res in results
    ]
    await asyncio.gather(*tasks)

def cli(
    url: str,
    instruction: str,
    output_dir: str = "~/Desktop/crawl_out",
    depth: int = 2,
    concurrency: int = 16,
    provider: str = "gemini/gemini-2.5-flash-preview-04-17",
    api_key: Optional[str] = None,
    md_min_chars: int = 1000
) -> None:
    """Thin wrapper that adapts Fire flags → `crawl_and_export` coroutine."""

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    asyncio.run(
        crawl_and_export(
            base_url=url,
            instruction=instruction,
            output_dir=out_path,
            depth=depth,
            model=provider,
            concurrency=concurrency,
            api_key=api_key,
            md_min_chars=md_min_chars
        )
    )

if __name__ == "__main__":
    fire.Fire(cli)

