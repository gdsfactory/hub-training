#!/bin/bash
set -e

# Minimal example: Upload two functions and create a pipeline using CLI

DATALAB_CLI="cargo run --manifest-path=../../Cargo.toml --bin gfhub --"

echo "Creating function scripts..."

# Function 1: CSV to Parquet
cat > /tmp/csv2parquet.py << 'EOF'
# /// script
# requires-python = ">=3.11"
# dependencies = ["pandas", "pyarrow"]
# ///

from pathlib import Path
import pandas as pd

def main(path: Path, /) -> Path:
    """Convert CSV to Parquet."""
    df = pd.read_csv(path)
    output = path.with_suffix(".parquet")
    df.to_parquet(output, index=False)
    return output
EOF

# Function 2: Parquet to JSON
cat > /tmp/parquet2json.py << 'EOF'
# /// script
# requires-python = ">=3.11"
# dependencies = ["pandas", "pyarrow"]
# ///

from pathlib import Path
import pandas as pd

def main(path: Path, /, *, orient: str = "records", indent: int = 2) -> Path:
    """Convert Parquet to JSON."""
    df = pd.read_parquet(path)
    output = path.with_suffix(".json")
    df.to_json(output, orient=orient, indent=indent)
    return output
EOF

echo "Creating functions..."
# update defaults to true, so we don't need --no-update
$DATALAB_CLI add-function csv2parquet /tmp/csv2parquet.py
$DATALAB_CLI add-function parquet2json /tmp/parquet2json.py

# Create pipeline YAML
cat > /tmp/csv2json_pipeline.yaml << 'EOF'
nodes:
  - name: on_file_upload
    type: on_file_upload
    config: {}
  - name: on_manual_trigger
    type: on_manual_trigger
    config: {}
  - name: load
    type: load
    config: {}
  - name: to_parquet
    type: function
    config:
      function: csv2parquet
      kwargs: {}
  - name: to_json
    type: function
    config:
      function: parquet2json
      kwargs:
        orient: records
        indent: 2
  - name: save
    type: save
    config: {}
edges:
  - source:
      node: on_file_upload
      output: 0
    target:
      node: load
      input: 0
  - source:
      node: on_manual_trigger
      output: 0
    target:
      node: load
      input: 0
  - source:
      node: load
      output: 0
    target:
      node: to_parquet
      input: 0
  - source:
      node: to_parquet
      output: 0
    target:
      node: to_json
      input: 0
  - source:
      node: to_json
      output: 0
    target:
      node: save
      input: 0
EOF

echo ""
echo "Creating pipeline..."
# update defaults to true, so we don't need --no-update
$DATALAB_CLI add-pipeline csv2json /tmp/csv2json_pipeline.yaml

echo ""
echo "✅ Done! Upload a CSV file to trigger the pipeline."

# Cleanup
rm /tmp/csv2parquet.py /tmp/parquet2json.py /tmp/csv2json_pipeline.yaml
