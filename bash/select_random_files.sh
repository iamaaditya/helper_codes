#!/bin/bash
for i in `seq 1 100`; do ls "${files[RANDOM % ${#files[@]}]}" ; done
