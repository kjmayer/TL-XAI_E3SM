#!/bin/bash

# Define the base directory containing monthly folders
BASE_DIR="/gpfs/csfs1/collections/rda/data/ds633.0/e5.oper.an.pl/"
# Define the directory to store daily mean files
DAILY_OUTPUT_BASE="/glade/derecho/scratch/kjmayer/DATA/ERA5/z500/"
# Define the final concatenated output file
FINAL_DAILY_OUTPUT="/glade/derecho/scratch/kjmayer/DATA/ERA5/z500/z500_daily_1996-2023.nc"

# Create the base directory for daily means if it doesn't exist
mkdir -p "$DAILY_OUTPUT_BASE"

# List to store all daily mean file paths
ALL_DAILY_MEANS=()

# Loop through each monthly folder
for MONTH_DIR in "$BASE_DIR"/*/; do
    # Extract the folder name (YYYYMM format)
    MONTH_NAME=$(basename "$MONTH_DIR")

    # Extract the year (first 4 characters of YYYYMM)
    YEAR=${MONTH_NAME:0:4}

    # Check if the year is within the range 1996-2023
    if [[ $YEAR -ge 2009 && $YEAR -le 2023 ]]; then
        DAILY_OUTPUT_DIR="$DAILY_OUTPUT_BASE/$MONTH_NAME"
        # Create a corresponding output directory for daily means
        mkdir -p "$DAILY_OUTPUT_DIR"
        # Process each daily file
        for FILE in "$MONTH_DIR"/e5.oper.an.pl.128_129_z.ll025sc.*.nc; do
            DAILY_MEAN_FILE="$DAILY_OUTPUT_DIR/$(basename "${FILE%.nc}_daily_mean.nc")"
            cdo daymean "$FILE" "$DAILY_MEAN_FILE"
            ALL_DAILY_MEANS+=("$DAILY_MEAN_FILE")
        done
    fi
done

# Merge all daily means across all selected years into a single file
cdo mergetime "${ALL_DAILY_MEANS[@]}" "$FINAL_DAILY_OUTPUT"