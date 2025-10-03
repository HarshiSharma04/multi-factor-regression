"""
Generate a pastel HTML report using the template in templates/.
"""
from jinja2 import Environment, FileSystemLoader
from pathlib import Path
import shutil
import os

def generate_html_report(out_path, title, summary, plot_paths, json_path):
    project_root = Path(__file__).resolve().parent.parent
    templates_dir = project_root / "templates"
    env = Environment(loader=FileSystemLoader(str(templates_dir)))
    template = env.get_template("pastel_report_template.html")
    # Map plot paths to relative paths for HTML (report will be in outputs/)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # Copy plot images into same folder as report for easy relative linking
    copied_plot_paths = []
    for plot in plot_paths:
        p = Path(plot)
        if p.exists():
            dest = out_path.parent / p.name
            shutil.copy(p, dest)
            copied_plot_paths.append(dest.name)
    # Copy JSON near report
    json_dest = out_path.parent / Path(json_path).name
    try:
        shutil.copy(json_path, json_dest)
    except Exception:
        json_dest = json_path  # fallback to original if copy fails

    html = template.render(title=title, summary=summary, plots=copied_plot_paths, json_path=json_dest.name)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html)
    return out_path
