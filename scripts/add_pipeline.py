#!/usr/bin/env uv run
"""Minimal example: Upload two functions and create a pipeline."""

from gfhub import Client, nodes

# Initialize client
client = Client()

# Function 1: CSV to Parquet
csv2parquet_script = '''
# @@@ script
# requires-python = ">=3.11"
# dependencies = ["pandas", "pyarrow"]
# @@@

from pathlib import Path
import pandas as pd

def main(path: Path, /) -> Path:
    """Convert CSV to Parquet."""
    df = pd.read_csv(path)
    output = path.with_suffix(".parquet")
    df.to_parquet(output, index=False)
    return output
'''.replace("@@@", "///")

print("Creating csv2parquet function...")
# update defaults to True
func1 = client.add_function("csv2parquet", csv2parquet_script)
print(f"✓ Created function: {func1['name']} (ID: {func1['id']})")

# Function 2: Parquet to JSON
parquet2json_script = '''
# @@@ script
# requires-python = ">=3.11"
# dependencies = ["pandas", "pyarrow"]
# @@@

from pathlib import Path
import pandas as pd

def main(path: Path, /, *, orient: str = "records", indent: int = 2) -> Path:
    """Convert Parquet to JSON."""
    df = pd.read_parquet(path)
    output = path.with_suffix(".json")
    df.to_json(output, orient=orient, indent=indent)
    return output
'''.replace("@@@", "///")

print("\nCreating parquet2json function...")
# update defaults to True
func2 = client.add_function("parquet2json", parquet2json_script)
print(f"✓ Created function: {func2['name']} (ID: {func2['id']})")

# Create cascaded pipeline
print("\nCreating cascaded pipeline...")
pipeline = client.add_cascaded_pipeline(
    name="csv2json",
    # Each layer connects to all nodes in the next layer
    layers=[
        [nodes.on_file_upload(), nodes.on_manual_trigger()],
        nodes.load(),
        nodes.function(function="csv2parquet", kwargs={}),
        nodes.function(
            "to_json",
            function="parquet2json",
            kwargs={"orient": "records", "indent": 2},
        ),
        nodes.save(),
    ],
)
print(f"✓ Created pipeline: {pipeline['name']} (ID: {pipeline['id']})")
print("\n✅ Done! Upload a CSV file to trigger the pipeline.")
