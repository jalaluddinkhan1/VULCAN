#!/usr/bin/env python3
"""
VULCAN Benchmark Visualization — Performance Graphs for README.

Reads JSON results from bench_inference and generates publication-quality plots:
  1. Kernel latency comparison (bar chart)
  2. GFLOPS performance (bar chart)
  3. Fused vs unfused kernel comparison
  4. Memory usage by model config

Usage:
    python plot_results.py --results benchmark_results.json --output plots/

"""

import argparse
import json
import os
import sys

def load_results(path):
    """Load benchmark results from JSON."""
    with open(path, 'r') as f:
        return json.load(f)

def generate_ascii_chart(results, metric='mean_us', title='Kernel Latency'):
    """Generate an ASCII bar chart for terminal output."""
    items = results.get('results', [])
    if not items:
        return

    max_val = max(r[metric] for r in items if r[metric] > 0)
    bar_width = 40

    print(f"\n  {title}")
    print(f"  {'─' * 72}")

    for r in items:
        val = r[metric]
        if val <= 0:
            continue
        bar_len = int((val / max_val) * bar_width)
        bar = '█' * bar_len + '░' * (bar_width - bar_len)
        name = r['name'][:28].ljust(28)
        print(f"  {name} │{bar}│ {val:.1f}")

    print(f"  {'─' * 72}")

def generate_html_report(results, output_dir):
    """Generate an HTML report with embedded SVG charts."""
    os.makedirs(output_dir, exist_ok=True)

    items = results.get('results', [])
    if not items:
        print("[PLOT] No results to plot.")
        return

    # Generate SVG bar chart for latency
    max_us = max(r['mean_us'] for r in items)
    chart_height = len(items) * 40 + 80
    bar_max_width = 400

    svg_lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 700 {chart_height}">',
        '<style>',
        '  text { font-family: "Consolas", monospace; font-size: 12px; fill: #e0e0e0; }',
        '  .title { font-size: 16px; font-weight: bold; fill: #ffffff; }',
        '  .bar { rx: 3; ry: 3; }',
        '  .value { font-size: 11px; fill: #b0b0b0; }',
        '</style>',
        '<rect width="700" height="{}" fill="#1a1a2e"/>'.format(chart_height),
        '<text x="350" y="30" text-anchor="middle" class="title">VULCAN Kernel Latency (μs)</text>',
    ]

    colors = ['#4ecdc4', '#ff6b6b', '#45b7d1', '#96ceb4', '#feca57',
              '#ff9ff3', '#54a0ff', '#5f27cd', '#01a3a4', '#2ed573']

    for i, r in enumerate(items):
        y = 50 + i * 40
        bar_width = int((r['mean_us'] / max_us) * bar_max_width)
        color = colors[i % len(colors)]
        name = r['name'][:30]

        svg_lines.extend([
            f'<text x="10" y="{y + 18}" class="label">{name}</text>',
            f'<rect x="250" y="{y}" width="{bar_width}" height="25" fill="{color}" class="bar" opacity="0.85"/>',
            f'<text x="{255 + bar_width}" y="{y + 18}" class="value">{r["mean_us"]:.1f} μs</text>',
        ])

    svg_lines.append('</svg>')

    svg_path = os.path.join(output_dir, 'kernel_latency.svg')
    with open(svg_path, 'w') as f:
        f.write('\n'.join(svg_lines))
    print(f"  [PLOT] Saved: {svg_path}")

    # Generate GFLOPS chart (only for entries with GFLOPS > 0)
    gflops_items = [r for r in items if r.get('gflops', 0) > 0]
    if gflops_items:
        max_gf = max(r['gflops'] for r in gflops_items)
        gf_height = len(gflops_items) * 40 + 80

        svg_gf = [
            f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 700 {gf_height}">',
            '<style>',
            '  text { font-family: "Consolas", monospace; font-size: 12px; fill: #e0e0e0; }',
            '  .title { font-size: 16px; font-weight: bold; fill: #ffffff; }',
            '  .bar { rx: 3; ry: 3; }',
            '  .value { font-size: 11px; fill: #b0b0b0; }',
            '</style>',
            f'<rect width="700" height="{gf_height}" fill="#1a1a2e"/>',
            '<text x="350" y="30" text-anchor="middle" class="title">VULCAN Compute Performance (GFLOPS)</text>',
        ]

        for i, r in enumerate(gflops_items):
            y = 50 + i * 40
            bar_width = int((r['gflops'] / max_gf) * bar_max_width)
            color = colors[(i + 3) % len(colors)]
            name = r['name'][:30]

            svg_gf.extend([
                f'<text x="10" y="{y + 18}" class="label">{name}</text>',
                f'<rect x="250" y="{y}" width="{bar_width}" height="25" fill="{color}" class="bar" opacity="0.85"/>',
                f'<text x="{255 + bar_width}" y="{y + 18}" class="value">{r["gflops"]:.1f} GFLOPS</text>',
            ])

        svg_gf.append('</svg>')

        svg_gf_path = os.path.join(output_dir, 'compute_performance.svg')
        with open(svg_gf_path, 'w') as f:
            f.write('\n'.join(svg_gf))
        print(f"  [PLOT] Saved: {svg_gf_path}")

    # Generate HTML report combining all charts
    html_path = os.path.join(output_dir, 'benchmark_report.html')
    with open(html_path, 'w') as f:
        f.write('<!DOCTYPE html>\n<html>\n<head>\n')
        f.write('<meta charset="utf-8">\n')
        f.write('<title>VULCAN Benchmark Report</title>\n')
        f.write('<style>body { background: #0d1117; color: #e0e0e0; ')
        f.write('font-family: "Consolas", monospace; max-width: 800px; ')
        f.write('margin: 0 auto; padding: 40px; }\n')
        f.write('h1, h2 { color: #4ecdc4; }\n')
        f.write('table { border-collapse: collapse; width: 100%; margin: 20px 0; }\n')
        f.write('th, td { padding: 8px 12px; border: 1px solid #333; text-align: right; }\n')
        f.write('th { background: #1a1a2e; color: #4ecdc4; }\n')
        f.write('</style>\n</head>\n<body>\n')
        f.write('<h1>⚡ VULCAN Benchmark Report</h1>\n')
        f.write(f'<p>Model: {results.get("model", "N/A")} | ')
        f.write(f'Precision: {results.get("precision", "N/A")}</p>\n')

        # Data table
        f.write('<h2>Kernel Results</h2>\n')
        f.write('<table>\n<tr><th>Kernel</th><th>Mean (μs)</th>')
        f.write('<th>Min (μs)</th><th>Max (μs)</th><th>GFLOPS</th></tr>\n')
        for r in items:
            gf = f'{r["gflops"]:.1f}' if r.get('gflops', 0) > 0 else '—'
            f.write(f'<tr><td style="text-align:left">{r["name"]}</td>')
            f.write(f'<td>{r["mean_us"]:.1f}</td>')
            f.write(f'<td>{r["min_us"]:.1f}</td>')
            f.write(f'<td>{r["max_us"]:.1f}</td>')
            f.write(f'<td>{gf}</td></tr>\n')
        f.write('</table>\n')

        # Embed SVG charts
        f.write('<h2>Latency Chart</h2>\n')
        f.write(f'<img src="kernel_latency.svg" width="700">\n')
        if gflops_items:
            f.write('<h2>Compute Performance</h2>\n')
            f.write(f'<img src="compute_performance.svg" width="700">\n')

        f.write('</body>\n</html>\n')

    print(f"  [PLOT] Saved: {html_path}")

def main():
    parser = argparse.ArgumentParser(description="VULCAN Benchmark Plotter")
    parser.add_argument("--results", type=str, required=True,
                        help="Path to results JSON")
    parser.add_argument("--output", type=str, default="plots/",
                        help="Output directory for charts")
    args = parser.parse_args()

    print("╔══════════════════════════════════════════════╗")
    print("║  VULCAN — Benchmark Visualization            ║")
    print("╚══════════════════════════════════════════════╝")
    print()

    results = load_results(args.results)

    # ASCII chart for terminal
    generate_ascii_chart(results, 'mean_us', 'Kernel Latency (μs)')

    gflops = [r for r in results.get('results', []) if r.get('gflops', 0) > 0]
    if gflops:
        generate_ascii_chart({'results': gflops}, 'gflops', 'Compute (GFLOPS)')

    # SVG + HTML report
    print(f"\n  Generating charts in {args.output}...")
    generate_html_report(results, args.output)

    print("\n[PLOT] Done.")

if __name__ == "__main__":
    main()
