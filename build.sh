#!/usr/bin/env bash

# Get the script's directory
SCRIPTPATH="$( cd "$(dirname "$0")" ; pwd -P )"

# Check for the 'build_base' argument
if [[ "$1" == "build_base" ]]; then
  echo "Rebuilding algorithm-base image..."
  docker build -t algorithm-base -f Dockerfile.base .
else
  # Check if algorithm-base image exists
  if [[ "$(docker images -q algorithm-base 2> /dev/null)" == "" ]]; then
    echo "Building algorithm-base image..."
    docker build -t algorithm-base -f Dockerfile.base .
  else
    echo "algorithm-base image already exists. Skipping build."
  fi
fi

# Build the surgtoolloc_det image
echo "Building surgtoolloc_det image..."
docker build -t surgtoolloc_det "$SCRIPTPATH"
