#!/bin/bash
# Run Triality comprehensive tests
# This script can be run from anywhere

# Get the directory where this script is located (triality/)
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Parent directory should be added to PYTHONPATH
PARENT_DIR="$(dirname "$SCRIPT_DIR")"

# Set PYTHONPATH to include the parent directory
export PYTHONPATH="$PARENT_DIR:$PYTHONPATH"

# Run the comprehensive test suite
python3 "$SCRIPT_DIR/test_all_features.py" "$@"
