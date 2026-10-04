project = "meTile"
copyright = "2026, Andre Slavescu"
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
html_logo = "_static/metile-logo.png"
html_favicon = "_static/metile-logo.png"
html_css_files = ["metile.css"]

# Furo theme options
html_theme_options = {
    "source_repository": "https://github.com/AndreSlavescu/meTile",
    "source_branch": "main",
    "source_directory": "docs/",
    "navigation_with_keys": True,
    "light_css_variables": {
        "color-brand-primary": "#086a8b",
        "color-brand-content": "#086a8b",
    },
    "dark_css_variables": {
        "color-brand-primary": "#65dad0",
        "color-brand-content": "#65dad0",
    },
}

# Syntax highlighting
pygments_style = "friendly"
pygments_dark_style = "monokai"
