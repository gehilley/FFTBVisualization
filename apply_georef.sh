#!/bin/bash

# Usage: ./apply_georef.sh source.tif target.tif output.tif
# Requires: jq (for parsing JSON)

SOURCE=$1
TARGET=$2
OUTPUT=$3
CONDA_ENV_NAME="py310"  # ← Change to your conda env name

if [ "$#" -ne 3 ]; then
    echo "Usage: $0 <source.tif> <target.tif> <output.tif>"
    exit 1
fi

# Activate conda environment
eval "$(conda shell.bash hook)"
conda activate "$CONDA_ENV_NAME"

# Check dependencies
command -v jq >/dev/null 2>&1 || { echo >&2 "jq is required but not installed. Exiting."; exit 1; }

# Get raster dimensions
SRC_SIZE=$(gdalinfo "$SOURCE" | grep 'Size is' | sed 's/[^0-9,]//g')
TGT_SIZE=$(gdalinfo "$TARGET" | grep 'Size is' | sed 's/[^0-9,]//g')

if [ "$SRC_SIZE" != "$TGT_SIZE" ]; then
    echo "❌ Dimensions do not match:"
    echo "Source: $SRC_SIZE"
    echo "Target: $TGT_SIZE"
    exit 2
fi

# Use JSON parsing to get corners and SRS
ULX=$(gdalinfo -json "$SOURCE" | jq '.cornerCoordinates.upperLeft[0]')
ULY=$(gdalinfo -json "$SOURCE" | jq '.cornerCoordinates.upperLeft[1]')
LRX=$(gdalinfo -json "$SOURCE" | jq '.cornerCoordinates.lowerRight[0]')
LRY=$(gdalinfo -json "$SOURCE" | jq '.cornerCoordinates.lowerRight[1]')
SRS=$(gdalsrsinfo -o epsg "$SOURCE" | grep EPSG)

echo "📍 Source SRS: $SRS"
echo "📐 UL: ($ULX, $ULY) | LR: ($LRX, $LRY)"

# Copy file and apply georef
cp "$TARGET" temp_georef.tif
gdal_edit.py -a_srs "$SRS" -a_ullr "$ULX" "$ULY" "$LRX" "$LRY" temp_georef.tif

mv temp_georef.tif "$OUTPUT"
echo "✅ Georeferencing applied successfully to $OUTPUT"

