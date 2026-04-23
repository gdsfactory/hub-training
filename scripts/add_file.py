#!/usr/bin/env uv run
"""Example script showing how to use gfhub to upload files."""

from pathlib import Path

from gfhub import Client

CWD = Path(__file__).parent.resolve()

client = Client()
path = Path(CWD.parent.parent / "assets" / "hub" / "files" / "lattice.gds")
# trigger_pipelines defaults to True, so we don't need to specify it
result = client.add_file(str(path), tags=["hey"])
print(result)
