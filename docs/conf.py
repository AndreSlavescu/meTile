project = "meTile"
copyright = "2025, Andre Slavescu"
author = "Andre Slavescu"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.viewcode",
    "sphinx_copybutton",
]

templates_path = ["_templates"]
exclude_patterns = ["_build"]

html_theme = "furo"
html_title = "meTile"
html_static_path = ["_static"]

# Furo theme options
html_theme_options = {
    "source_repository": "https://github.com/AndreSlavescu/meTile",
    "source_branch": "main",
    "source_directory": "docs/",
    "navigation_with_keys": True,
}

# Syntax highlighting
pygments_style = "friendly"
pygments_dark_style = "monokai"
