#!/bin/bash
for file in $(find . -name "*.csv.gz"); do
    # Decompress the file in place
    gunzip -f $file
done

for file in $(find . -name "*.json.gz"); do
    # Decompress the file in place
    gunzip -f $file
done