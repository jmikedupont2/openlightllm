#!/bin/bash

# # try except this script
# set -e

# print current dir 
echo
pwd


echo "Building Custom Admin UI..."

#curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.38.0/install.sh | bash
#source ~/.nvm/nvm.sh
#nvm install v18.17.0
#nvm use v18.17.0
#npm install -g npm

# cd in to /ui/litellm-dashboard
cd ui/litellm-dashboard

# ensure have access to build_ui.sh
chmod +x ./build_ui.sh

# run ./build_ui.sh
./build_ui.sh

# return to root directory
cd ../..
