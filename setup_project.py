import os

# Define the folder and file structure
project_structure = {
    "project-root": {
        "data": {
            "raw": {
                "audio_chunks": {}
            },
            "processed": {
                "features": {}
            },
            "playlists": {}
        },
        "notebooks": {
            "analysis_overview.ipynb": "",
            "demo_playlist_generation.ipynb": ""
        },
        "src": {
            "__init__.py": "",
            "audio_analysis.py": "",
            "feature_extraction.py": "",
            "similarity.py": "",
            "utils.py": "",
            "config.py": ""
        },
        "apps": {
            "descriptor_based_ui.py": "",
            "similarity_based_ui.py": ""
        },
        "reports": {
            "figures": {},
            "final_report.md": ""
        },
        "environment.yml": "",
        "requirements.txt": "",
        "README.md": "",
        ".gitignore": ""
    }
}

def create_structure(base_path, structure):
    """Recursively create folders and files based on the given dictionary structure."""
    for name, content in structure.items():
        path = os.path.join(base_path, name)
        
        if isinstance(content, dict):  # If it's a dictionary, create a folder
            os.makedirs(path, exist_ok=True)
            create_structure(path, content)  # Recursively create subdirectories
        else:  # If it's a string (empty), create a file
            with open(path, "w", encoding="utf-8") as f:
                f.write(content)

# Get current working directory and create project
base_directory = os.getcwd()
create_structure(base_directory, project_structure)

print("Proyecto creado exitosamente en:", os.path.join(base_directory, "project-root"))
