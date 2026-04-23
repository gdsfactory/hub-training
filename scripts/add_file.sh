#!/bin/sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SDK_DIR="$(dirname "$SCRIPT_DIR")"
CLI="$SDK_DIR/target/release/gfhub"

FILE="$SDK_DIR/../assets/hub/files/lattice.gds"

cd $SDK_DIR
cargo run -- --url http://localhost:8080 add-file "$FILE" --tags hoi test wafer_id:wafer1 die:3,2
