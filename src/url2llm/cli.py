from __future__ import annotations

import asyncio
import functools
import os
import re
from pathlib import Path
from typing import List, Optional

def _slugify(text: str) -> str:
    """Convert *text* to a URL-safe slug (max 120 chars)."""
    pattern = re.compile(r"[^\w-]+", re.UNICODE)
    slug = pattern.sub("-", text.strip().lower())
    return slug[:120].strip("-") or "merged"


async def _filter_content(llm_filter, content: str) -> List[str]:
    """Run the LLM content filter in a thread pool (non-blocking)."""
    loop = asyncio.get_running_loop()
    func = functools.partial(llm_filter.filter_content, content)
    return await loop.run_in_executor(None, func)


async def _process_page(
    result,
    llm_filter,
    semaphore: asyncio.Semaphore,
    min_chars: int,
) -> Optional[Path]:
    """Save a single crawled page if it survives filtering."""
    async with semaphore:
        if not result.success or not getattr(result, "markdown", None):
            return None

        raw_md = getattr(result.markdown, "raw_markdown", None) or (
            result.markdown if isinstance(result.markdown, str) else None
        )
        if not raw_md:
            return None

        chunks = await _filter_content(llm_filter, raw_md)
        md = "\n\n".join(chunks).strip()
        if len(md) < min_chars:
            return None

        filename = f"{_slugify(result.url)}.md"
        path = Path("/tmp") / "url2llm_pages" / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(md, encoding="utf-8")
        print(f"Saved → {path}")
        return path


async def _fetch_urls_from_llms_txt(url: str) -> List[str]:
    """Download a `llms.txt` and pull out all markdown-style links."""
    import aiohttp

    async with aiohttp.ClientSession() as sess:
        async with sess.get(url) as resp:
            if resp.status != 200:
                print(f"Failed to fetch llms.txt ({resp.status})")
                return []
            text = await resp.text()
            return re.findall(r"\]\((https?://[^)]+)\)", text)


async def _generate_title(
    filenames: List[str], provider: str, api_key: Optional[str]
) -> str:
    """Ask the LLM for a short merged-document title."""
    import litellm

    prompt = (
        "Based on these markdown filenames, generate a single, descriptive title "
        "for the merged document (≤50 chars, no extensions or punctuation):\n\n"
        + os.linesep.join(f"- {fn}" for fn in filenames)
        + "\n\nTitle:"
    )

    try:
        print(f"Using model → {provider} …")
        resp = await litellm.acompletion(
            model=provider,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            api_key=api_key,
        )
        title = resp.choices[0].message.content.strip()
        return _slugify(title) if title and len(title) >= 3 else "merged-content"
    except Exception as exc:  # noqa: BLE001
        print(f"Title generation failed: {exc}")
        return "merged-content"


# ────────────────────────────────────────────────
#  Core crawling pipeline (imports crawl4ai lazily)
# ────────────────────────────────────────────────
async def _crawl_website(
    *,
    url: str,
    instruction: str,
    depth: int,
    provider: str,
    concurrency: int,
    api_key: Optional[str],
    min_chars: int,
) -> List[Path]:
    """Crawl a website and save filtered pages as markdown."""
    # heavy imports here ↓
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

    llm_cfg = LLMConfig(provider=provider, api_token=api_key)
    llm_filter = LLMContentFilter(
        llm_config=llm_cfg,
        instruction=(
            f"{instruction}\n\n# Important rules:\n"
            "- IF the page is IRRELEVANT → return empty string.\n"
            "- ELSE omit nav menus, footers, cookie banners; keep headings, code, lists.\n"
            "- Return clean markdown only."
        ),
        chunk_token_threshold=8192,
        verbose=True,
    )

    md_gen = DefaultMarkdownGenerator(options={"ignore_links": True})
    deep_crawl = BFSDeepCrawlStrategy(max_depth=depth, include_external=False)
    run_cfg = CrawlerRunConfig(
        deep_crawl_strategy=deep_crawl,
        markdown_generator=md_gen,
        cache_mode=CacheMode.BYPASS,
        semaphore_count=concurrency,
        verbose=True,
    )

    start_urls = (
        await _fetch_urls_from_llms_txt(url)
        if url.lower().endswith("llms.txt")
        else [url]
    )

    results = []
    async with AsyncWebCrawler(config=BrowserConfig(headless=True)) as crawler:
        for u in start_urls:
            print(f"Crawling {u} (depth={depth}) …")
            results.extend(await crawler.arun(u, config=run_cfg))

    sem = asyncio.Semaphore(concurrency)
    tasks = [_process_page(r, llm_filter, sem, min_chars) for r in results]
    return [p for p in await asyncio.gather(*tasks) if p]


async def _merge_files(files: List[Path], output_dir: Path, filename: str, keep_pages: bool) -> None:
    """Concatenate individual markdown files into one."""
    output_dir.mkdir(parents=True, exist_ok=True)
    merged_path = output_dir / filename

    # Create merged file
    with merged_path.open("w", encoding="utf-8") as out:
        for i, path in enumerate(sorted(files), 1):
            out.write(
                f"\n{'*' * 80}\n** Section {i}: {path.name} **\n{'*' * 80}\n\n"
            )
            out.write(path.read_text(encoding="utf-8"))
            if i < len(files):
                out.write("\n\n")

    # Move page files or delete
    if keep_pages:
        pages_out_dir = output_dir / "url2llm_pages/"
        pages_out_dir.mkdir(parents=True, exist_ok=True)
        for p in files:
            try:
                p.rename(pages_out_dir / p.name)
            except Exception:
                print(f"Failed to move {p}. Skipping.")
        print(f"Kept pages → {pages_out_dir}")
    else:
        for p in files:
            try:
                p.unlink()
            except Exception:
                pass
        files[0].parent.rmdir()

    print(f"Merged {len(files)} files → {merged_path}")


async def _main_async(
    url: str,
    instruction: str,
    output_dir: Path,
    depth: int,
    provider: str,
    concurrency: int,
    api_key: Optional[str],
    min_chars: int,
    keep_pages: bool,
) -> None:
    """Async Orchestrator."""
    pages = await _crawl_website(
        url=url,
        instruction=instruction,
        depth=depth,
        provider=provider,
        concurrency=concurrency,
        api_key=api_key,
        min_chars=min_chars,
    )
    if not pages:
        print("No relevant pages found.")
        return

    title = await _generate_title([p.name for p in pages], provider, api_key)

    await _merge_files(pages, output_dir, f"{title}.md", keep_pages)


# ────────────────────────────────────────────────
# Public CLI wrapper (imports Fire lazily too)
# ────────────────────────────────────────────────
def _crawl_command(
    url: str,
    instruction: str,
    output_dir: str = ".",
    depth: int = 2,
    concurrency: int = 16,
    provider: str = "gemini/gemini-2.5-flash-preview-04-17",
    api_key: Optional[str] = None,
    min_chars: int = 1000,
    keep_pages: bool | str = False,
) -> None:
    """
    Crawl a website (or llms.txt) and emit LLM-ready markdown.

    :param url: Start URL or path to a `llms.txt`.
    :param instruction: LLM filtering instruction.
    :param output_dir: Where to dump markdown files (~/Desktop/… by default).
    :param depth: Link-depth to follow.
    :param concurrency: Parallel fetch/filter tasks.
    :param provider: LLM provider/model (e.g. "openai/gpt-4o").
    :param api_key: API key for the provider.
    :param min_chars: Discard filtered chunks shorter than this.
    :param keep_pages: Also keep individual pages, not only merged file
    """
    if isinstance(keep_pages, str):
        if keep_pages.lower() == "false":
            keep_pages = False
        elif keep_pages.lower() == "true":
            keep_pages = True
        else:
            raise ValueError("--keep_pages must one of: `true`, `False`")

    path = Path(output_dir).expanduser()
    asyncio.run(
        _main_async(
            url=url,
            instruction=instruction,
            output_dir=path,
            depth=depth,
            provider=provider,
            concurrency=concurrency,
            api_key=api_key,
            min_chars=min_chars,
            keep_pages=keep_pages,
        )
    )


def url2llm() -> None:
    import fire
    fire.Fire(_crawl_command)


if __name__ == "__main__":
    url2llm()
