"""
Crawl a website and export topic-focused markdown files, then merge them 
with an LLM-generated filename based on content.
"""

import asyncio
import functools
import os
import re
from pathlib import Path
from typing import List, Optional

import aiohttp
import fire
import litellm
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

_SLUG_PATTERN = re.compile(r"[^\w-]+", re.UNICODE)

def slugify(text: str) -> str:
    """Convert text to URL-safe slug, limited to 120 chars."""
    slug = _SLUG_PATTERN.sub("-", text.strip().lower())
    return slug[:120].strip("-") or "merged"

async def filter_content(llm_filter, content: str) -> List[str]:
    """Filter content using LLM"""
    loop = asyncio.get_running_loop()
    filter_func = functools.partial(llm_filter.filter_content, content)
    return await loop.run_in_executor(None, filter_func)

async def process_page(
    result,
    llm_filter,
    output_dir: Path,
    semaphore: asyncio.Semaphore,
    min_chars: int
) -> Optional[Path]:
    """Process a single crawled page and save as markdown if relevant."""
    async with semaphore:
        # Skip failed or empty results
        if not result.success or not getattr(result, "markdown", None):
            return None

        # Extract raw markdown content
        raw_md = getattr(result.markdown, "raw_markdown", None) or (
            result.markdown if isinstance(result.markdown, str) else None
        )
        if not raw_md:
            return None

        # Filter content using LLM
        chunks = await filter_content(llm_filter, raw_md)
        md = "\n\n".join(chunks).strip()

        # Skip if content is too short
        if len(md) < min_chars:
            return None

        # Save to file
        filename = f"{slugify(result.url)}.md"
        path = output_dir / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(md, encoding="utf-8")
        print(f"Saved → {path}")

        return path

async def fetch_urls_from_llms_txt(url: str) -> List[str]:
    """Parse a llms.txt file and extract URLs."""
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as resp:
            if resp.status != 200:
                print(f"Failed to fetch llms.txt: {resp.status}")
                return []
            text = await resp.text()
            # Extract markdown-style URLs
            return re.findall(r"\]\((https?://[^)]+)\)", text)

async def generate_title(filenames: List[str], provider: str, api_key: Optional[str]) -> str:
    """Generate a descriptive title using the llm."""
    prompt = f"""Based on these markdown filenames, generate a single, descriptive title for a document that merges all of them.
The title should be concise (under 50 characters) and capture the main topic or theme.
Do not include file extensions or special characters in the title.
Just return the title text with no additional explanation or formatting.

Files:
{os.linesep.join(f"- {filename}" for filename in filenames)}

Title:"""

    try:
        print(f"Using model: {provider} for title generation")

        response = await litellm.acompletion(
            model=provider,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            api_key=api_key
        )
        title = response.choices[0].message.content.strip()

        print(f"Generated title: '{title}'")

        # If empty or too short, use fallback
        if not title or len(title) < 3:
            print("Warning: Generated title was too short, using fallback")
            return "merged-content"

        # Slugify the title and return
        return slugify(title)

    except Exception as e:
        print(f"Error generating title: {e}")
        return "merged-content"

async def crawl_website(
    url: str,
    instruction: str,
    output_dir: Path,
    depth: int,
    model: str,
    concurrency: int,
    api_key: Optional[str],
    min_chars: int
) -> List[Path]:
    """Crawl website and export pages as markdown files."""
    # Setup LLM filter with instruction
    llm_cfg = LLMConfig(provider=model, api_token=api_key)
    llm_filter = LLMContentFilter(
        llm_config=llm_cfg,
        instruction=f"""
{instruction}

# Important rules:
- IF the page is IRRELEVANT with respect to the main instruction
    - skip the page and return an empty string.
- ELSE IF the page is RELEVANT with respect to the main instruction:
    - Omit navigation menus, sidebars, footers, cookie banners, ads.
    - Keep headings, lists, code blocks, tables and formatting intact.
    - Return clean, well-structured markdown only.
""",
        chunk_token_threshold=8192,
        verbose=True,
    )

    # Configure crawler
    md_gen = DefaultMarkdownGenerator(options={"ignore_links": True})
    deep_crawl = BFSDeepCrawlStrategy(max_depth=depth, include_external=False)
    run_cfg = CrawlerRunConfig(
        deep_crawl_strategy=deep_crawl,
        markdown_generator=md_gen,
        cache_mode=CacheMode.BYPASS,
        semaphore_count=concurrency,
        verbose=True,
    )

    # Determine start URLs
    start_urls = (await fetch_urls_from_llms_txt(url) 
                  if url.lower().endswith('llms.txt') 
                  else [url])

    results = []
    async with AsyncWebCrawler(config=BrowserConfig(headless=True)) as crawler:
        for url in start_urls:
            print(f"Crawling {url} (depth={depth})...")
            batch = await crawler.arun(url, config=run_cfg)
            results.extend(batch)

    # Process pages concurrently
    sem = asyncio.Semaphore(concurrency)
    tasks = [process_page(r, llm_filter, output_dir, sem, min_chars) for r in results]
    return [p for p in await asyncio.gather(*tasks) if p]

async def merge_files(input_paths: List[Path], output_file: Path) -> None:
    """Merge multiple markdown files into one."""
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with output_file.open('w', encoding='utf-8') as out:
        for i, path in enumerate(sorted(input_paths), 1):
            out.write(f"\n{'*'*80}"
                      f"\n** Section {i}: {path.name} **"
                      f"\n{'*'*80}"
                      f"\n\n")
            out.write(path.read_text(encoding='utf-8'))

            # Add separator between sections
            if i < len(input_paths):
                out.write("\n\n")

    print(f"Merged {len(input_paths)} files into {output_file}")

async def main_async(
    url: str,
    instruction: str,
    output_dir: Path,
    depth: int,
    provider: str,
    concurrency: int,
    api_key: Optional[str],
    min_chars: int
) -> None:
    """Main async function to orchestrate the crawling and merging process."""
    output_dir.mkdir(parents=True, exist_ok=True)

    paths = await crawl_website(
        url=url,
        instruction=instruction,
        output_dir=output_dir,
        depth=depth,
        model=provider,
        concurrency=concurrency,
        api_key=api_key,
        min_chars=min_chars
    )

    if not paths:
        print("No relevant pages found to merge.")
        return

    title = await generate_title(
        [p.name for p in paths], 
        provider, 
        api_key
    )

    merged_file = output_dir / "merged" / f"{title}.md"
    await merge_files(paths, merged_file)

def cli(
    url: str,
    instruction: str,
    output_dir: str = "~/Desktop/crawl_out",
    depth: int = 2,
    concurrency: int = 16,
    provider: str = "gemini/gemini-2.5-flash-preview-04-17",
    api_key: Optional[str] = None,
    min_chars: int = 1000
) -> None:
    """Command-line interface for web crawling and markdown generation."""
    out_path = Path(output_dir).expanduser()

    asyncio.run(main_async(
        url=url,
        instruction=instruction,
        output_dir=out_path,
        depth=depth,
        provider=provider,
        concurrency=concurrency,
        api_key=api_key,
        min_chars=min_chars
    ))

if __name__ == "__main__":
    fire.Fire(cli)
