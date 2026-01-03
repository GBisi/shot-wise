#!/bin/bash
# Copyright 2024 Shot-wise contributors
# Licensed under the Apache License, Version 2.0. See the LICENSE file for details.

# Find all .csv files in all subdirectories
for file in $(find . -name "*.csv"); do
    # Compress the file in place
    gzip -f "$file" -9
done

# Find all .json files in all subdirectories
for file in $(find . -name "*.json"); do
    # Compress the file in place
    gzip -f "$file" -9
done
