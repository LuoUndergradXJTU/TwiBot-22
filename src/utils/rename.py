import glob
import os
from pathlib import Path

md_files = glob.glob("../*/*.md")

for md_file in md_files:
    new_parts = list(Path(md_file).parts[:-1]) + ["readme.md"]
    new_file_name = Path(*new_parts)
    os.rename(md_file, new_file_name)
