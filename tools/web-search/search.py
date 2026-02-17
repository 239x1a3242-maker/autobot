#!/usr/bin/env python3
"""
Complete search pipeline wrapper.
Runs quick_scrape.py → main_content_cleaner.py
Outputs: structured JSON with filtered results

Usage (CLI):
  python search.py --query "today hot news" --max 50 --workers 6 --out results.json

This imports `EnterpriseSearchEngine` from `quick_scrape.py` and `process_results` from `main_content_cleaner.py`
"""
import argparse
import json
from dataclasses import asdict
from pathlib import Path

try:
    from quick_scrape import EnterpriseSearchEngine
    from main_content_cleaner import process_results
except Exception as e:
    raise ImportError("Could not import from quick_scrape.py or main_content_cleaner.py: " + str(e))


def run_search(query: str, max_results: int = 100, workers: int = 8):
    """
    Run complete search pipeline: search → clean → filter.
    
    Args:
        query: Search query string
        max_results: Max results to fetch
        workers: Parallel workers
    
    Returns:
        (structured_results, stats) tuple
    """
    # Phase 1: Search and extract
    engine = EnterpriseSearchEngine(max_workers=workers)
    raw_results = engine.execute_search(query, max_results)
    print(f"📈 Total chars in raw_results( from search.py): {len(raw_results)}")
    # Convert dataclass results to plain dicts
    results_dicts = [asdict(r) for r in raw_results]
    
    # Phase 2: Clean and filter by extraction_status == "success"
    structured_results, cleaner_stats = process_results(results_dicts)
    
    # Combine stats
    combined_stats = {
        'search_engine': engine.stats,
        'cleaner': cleaner_stats
    }
    print(f"📈 Total chars in extraction_status( from search.py): {len(structured_results)}")
    return structured_results, combined_stats


def main():
    parser = argparse.ArgumentParser(description='Complete search pipeline: search → clean → filter')
    parser.add_argument('--query', '-q', required=True, help='Search query')
    parser.add_argument('--max', '-m', type=int, default=100, help='Max results')
    parser.add_argument('--workers', '-w', type=int, default=8, help='Parallel workers')
    parser.add_argument('--out', '-o', default='struct_format_results.json', help='Output structured JSON path')

    args = parser.parse_args()

    # Run full pipeline
    structured_results, stats = run_search(args.query, max_results=args.max, workers=args.workers)

    # Build final output
    output = {
        'query': args.query,
        'parameters': {'max_results': args.max, 'workers': args.workers},
        'stats': stats,
        'structured_results': structured_results
    }

    # Save structured JSON
    out_path = Path(args.out)
    out_path.write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding='utf-8')
    
    print(f'\n✅ SEARCH PIPELINE COMPLETE!')
    print(f'   🔍 Query: {args.query}')
    print(f'   📊 Total results from search: {stats["search_engine"]["total"]}')
    print(f'   ✅ Successfully extracted: {stats["cleaner"]["successful"]}')
    print(f'   ❌ Failed (ignored): {stats["cleaner"]["failed"]}')
    print(f'   📄 Structured JSON: {out_path}')
    print(f'   ⏱️  Execution time: {stats["search_engine"]["execution_time"]:.1f}s')


if __name__ == '__main__':
    main()

