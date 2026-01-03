#!/bin/bash
# Copyright 2024 Shot-wise contributors
# Licensed under the Apache License, Version 2.0. See the LICENSE file for details.

for file in $(find . -name "*.csv.gz"); do
    # Decompress the file in place
    gunzip -f "$file"
done

for file in $(find . -name "*.json.gz"); do
    # Decompress the file in place
    gunzip -f "$file"
done
