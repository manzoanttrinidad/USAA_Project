#!/usr/bin/env python3
import argparse
import csv
import gzip
import io
import re
import sys
from urllib.parse import urljoin, urlparse

import requests
from xml.etree import ElementTree as ET

HEADERS = {
    "User-Agent": "SitemapURLCollector/1.0 (+https://example.com; contact=you@example.com)"
}

def fetch_bytes(url, timeout=20):
    """GET bytes; transparently handle .gz resources."""
    r = requests.get(url, headers=HEADERS, timeout=timeout, stream=True)
    r.raise_for_status()
    data = r.content
    # If URL ends with .gz or server sends gzip content-type, decompress
    ctype = r.headers.get("Content-Type", "").lower()
    if url.lower().endswith(".gz") or "gzip" in ctype:
        try:
            data = gzip.decompress(data)
        except OSError:
            # Some servers already decompress but keep .gz URL—fall back gracefully
            pass
    return data

def robots_sitemaps(site_root):
    """
    Read robots.txt and return list of 'Sitemap:' URLs.
    site_root can be 'https://example.com' or any page on that domain.
    """
    parsed = urlparse(site_root)
    base = f"{parsed.scheme}://{parsed.netloc}"
    robots_url = urljoin(base, "/robots.txt")
    try:
        text = fetch_bytes(robots_url).decode("utf-8", errors="replace")
    except Exception:
        return []
    sitemaps = []
    for line in text.splitlines():
        line = line.strip()
        # Format: Sitemap: https://example.com/sitemap.xml
        if line.lower().startswith("sitemap:"):
            sm_url = line.split(":", 1)[1].strip()
            sitemaps.append(sm_url)
    return sitemaps

def parse_xml_for_entries(xml_bytes):
    """
    Parse either a <urlset> (URLs) or <sitemapindex> (nested sitemaps).
    Returns two lists: (urls, nested_sitemaps)
    Each url is dict with keys: 'loc', 'lastmod' (optional).
    """
    # Remove namespace prefixes to make tag matching easy
    it = ET.iterparse(io.BytesIO(xml_bytes))
    for _, el in it:
        if "}" in el.tag:
            el.tag = el.tag.split("}", 1)[1]
    root = it.root

    urls = []
    nested = []

    if root.tag.lower() == "urlset":
        for url_el in root.findall("url"):
            loc_el = url_el.find("loc")
            if loc_el is None or not (loc_el.text and loc_el.text.strip()):
                continue
            entry = {"loc": loc_el.text.strip()}
            lastmod_el = url_el.find("lastmod")
            if lastmod_el is not None and lastmod_el.text:
                entry["lastmod"] = lastmod_el.text.strip()
            urls.append(entry)
    elif root.tag.lower() == "sitemapindex":
        for sm_el in root.findall("sitemap"):
            loc_el = sm_el.find("loc")
            if loc_el is None or not (loc_el.text and loc_el.text.strip()):
                continue
            nested.append(loc_el.text.strip())
    else:
        # Some sites serve HTML by mistake—try to scrape <loc> anywhere
        for loc_el in root.iter("loc"):
            if loc_el.text and loc_el.text.strip():
                urls.append({"loc": loc_el.text.strip()})
    return urls, nested

def crawl_sitemaps(seed_sitemaps, max_sitemaps=None, verbose=False):
    """
    Breadth-first expansion of sitemap indexes -> urlsets.
    Returns list of URL dicts: {'loc': ..., 'lastmod': optional}
    """
    seen_sitemaps = set()
    queue = list(seed_sitemaps)
    results = []
    while queue:
        sm_url = queue.pop(0)
        if sm_url in seen_sitemaps:
            continue
        seen_sitemaps.add(sm_url)

        if verbose:
            print(f"[SITEMAP] {sm_url}", file=sys.stderr)
        try:
            xml_bytes = fetch_bytes(sm_url)
        except Exception as e:
            if verbose:
                print(f"  ! fetch error: {e}", file=sys.stderr)
            continue

        try:
            urls, nested = parse_xml_for_entries(xml_bytes)
        except Exception as e:
            if verbose:
                print(f"  ! parse error: {e}", file=sys.stderr)
            continue

        if nested:
            # sitemap index: enqueue children
            for child in nested:
                if child not in seen_sitemaps:
                    queue.append(child)
        if urls:
            results.extend(urls)

        if max_sitemaps and len(seen_sitemaps) >= max_sitemaps:
            break

    # Deduplicate by loc, prefer first seen lastmod
    dedup = {}
    for u in results:
        loc = u["loc"].strip()
        if loc not in dedup:
            dedup[loc] = u
    return list(dedup.values())

def guess_default_sitemaps(site_root):
    """
    If robots.txt didn't list any, try a few common sitemap paths.
    """
    parsed = urlparse(site_root)
    base = f"{parsed.scheme}://{parsed.netloc}"
    candidates = [
        "/sitemap.xml",
        "/sitemap_index.xml",
        "/sitemap-index.xml",
        "/sitemap.gz",
        "/sitemap_index.xml.gz",
    ]
    out = []
    for c in candidates:
        url = urljoin(base, c)
        try:
            b = fetch_bytes(url)
            # Quick sanity: it should look like XML
            if b.strip().startswith(b"<?xml") or b[:100].lower().find(b"<urlset") != -1 or b[:100].lower().find(b"<sitemapindex") != -1:
                out.append(url)
        except Exception:
            pass
    return out

def write_out(urls, path, fmt):
    if fmt == "csv":
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["url", "lastmod"])
            for u in urls:
                w.writerow([u.get("loc", ""), u.get("lastmod", "")])
    else:
        with open(path, "w", encoding="utf-8") as f:
            for u in urls:
                f.write(u.get("loc", "") + "\n")

def main():
    ap = argparse.ArgumentParser(description="Collect URLs from a site's sitemaps")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--site", help="Site root like https://example.com (will read robots.txt)")
    g.add_argument("--sitemap", action="append", help="Direct sitemap URL (repeatable)")
    ap.add_argument("--max-sitemaps", type=int, default=None, help="Limit number of sitemaps processed (for testing)")
    ap.add_argument("--out", default="urls.csv", help="Output file path")
    ap.add_argument("--format", choices=["csv", "txt"], default="csv", help="Output format")
    ap.add_argument("-v", "--verbose", action="store_true", help="Verbose progress to stderr")
    args = ap.parse_args()

    if args.sitemap:
        seeds = args.sitemap
    else:
        seeds = robots_sitemaps(args.site)
        if args.verbose:
            print(f"[ROBOTS] found {len(seeds)} sitemap(s)", file=sys.stderr)
        if not seeds:
            # Try common default paths
            guesses = guess_default_sitemaps(args.site)
            if args.verbose:
                print(f"[GUESS] found {len(guesses)} probable sitemap(s)", file=sys.stderr)
            seeds = guesses

    if not seeds:
        print("No sitemaps discovered. Provide --sitemap or check the site.", file=sys.stderr)
        sys.exit(2)

    urls = crawl_sitemaps(seeds, max_sitemaps=args.max_sitemaps, verbose=args.verbose)
    # Sort by lastmod desc when available, then by URL
    def sort_key(u):
        return (u.get("lastmod") or "", u.get("loc") or "")
    urls.sort(key=sort_key, reverse=True)

    write_out(urls, args.out, args.format)
    print(f"Collected {len(urls)} URL(s) → {args.out}")

if __name__ == "__main__":
    main()
