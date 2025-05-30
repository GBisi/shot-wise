#!/bin/bash

# Find all .csv files in all subdirectories
for file in $(find . -name "*.csv"); do
    # Compress the file in place
    gzip -f $file -9
done

# Find all .json files in all subdirectories
for file in $(find . -name "*.json"); do
    # Compress the file in place
    gzip -f $file -9
done