#!/bin/bash

while [[ $# -gt 0 ]]; do
    case $1 in 
    --checkpoint_path)
        checkpoint_path=$2
        ;;
    esac
    shift
done



function prepare(){
    echo "Getting checkpoint..."
    cp $checkpoint_path .

    echo "Copied from ${checkpoint_path}"

    if ! [[ $(ls | grep -q nutcl) ]]; then
        echo "Downloading nuctl..."

        curl -s https://api.github.com/repos/nuclio/nuclio/releases/latest \
			| grep -i "browser_download_url.*nuctl.*$(uname)" \
			| cut -d : -f 2,3 \
			| tr -d \" \
			| wget -O nuctl -qi - && chmod +x nuctl
    fi

    if  [[ $(./nuctl get projects --platform local | grep -q cvat) ]]; then
        echo "Creating CVAT project..."
        ./nuctl create project cvat --platform local
    fi
}

function deploy(){
    echo "Deploying nuclio..."
    ./nuctl deploy --project-name cvat --path . --volume `pwd`/../../src:/opt/nuclio/src --platform local
}

# prepare
deploy