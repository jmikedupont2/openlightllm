#!/bin/bash


# print contents of ui_colors.json
echo "Contents of ui_colors.json:"
cat ui_colors.json

npm install .

# Run npm build
npm run build

# Check if the build was successful
if [ $? -eq 0 ]; then
  echo "Build successful. Copying files..."

  # echo current dir
  echo
  pwd

  # Specify the destination directory
  destination_dir="../../litellm/proxy/_experimental/out"

  # Remove existing files in the destination directory
  rm -rf "$destination_dir"/*

  # Copy the contents of the output directory to the specified destination
  cp -r ./out/* "$destination_dir"

  echo "Deployment completed."
else
  echo "Build failed. Deployment aborted."
fi
