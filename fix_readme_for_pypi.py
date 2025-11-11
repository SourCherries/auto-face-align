import re
from pathlib import Path

# Adjust these if your repo changes name or branch
GITHUB_USER = "SourCherries"
REPO_NAME = "auto-face-align"
BRANCH = "main"

readme_path = Path("README.md")
output_path = Path("README_PyPI.md")

if not readme_path.exists():
    raise FileNotFoundError("Could not find README.md")

text = readme_path.read_text(encoding="utf-8")

# Convert image paths like ![Alt](images/foo.png)
text = re.sub(
    r'!\[([^\]]*)\]\((?!https?://)([^)]+)\)',
    lambda m: f'![{m.group(1)}](https://github.com/{GITHUB_USER}/{REPO_NAME}/raw/{BRANCH}/{m.group(2)})',
    text,
)

# Convert markdown links like [label](docs/demo.md)
text = re.sub(
    r'\[([^\]]+)\]\((?!https?://)([^)]+)\)',
    lambda m: f'[{m.group(1)}](https://github.com/{GITHUB_USER}/{REPO_NAME}/tree/{BRANCH}/{m.group(2)})',
    text,
)

output_path.write_text(text, encoding="utf-8")

print(f"✅ Created {output_path} — safe for PyPI upload.")
