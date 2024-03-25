#!/bin/bash

function download_checkpoint(){
    if [[ ! -z $checkpoint_path ]]; then
        if [[ ! -f $(pwd)"/checkpoint.pth" ]]; then
            echo "Getting checkpoint from ${checkpoint_path}"
            cp $checkpoint_path ./checkpoint.pth
        else
            echo "Checkpoint already exists. Skipping download."
        fi
    else
        echo "Checkpoint path not specified. Skipping download."
    fi
}

function download_nuclio(){
    if [[ ! -f ./nuctl ]]; then
        echo "Downloading nuctl..."

        curl -s https://api.github.com/repos/nuclio/nuclio/releases/latest \
			| grep -i "browser_download_url.*nuctl.*$(uname)" \
			| cut -d : -f 2,3 \
			| tr -d \" \
			| wget -O nuctl -qi - && chmod +x nuctl
    else
        echo "nuctl already exists. Skipping download."
    fi

}

function create_cvat_project(){
    if [[ ! -z $(./nuctl get projects --platform local | grep -q cvat) ]]; then
        echo "Creating CVAT project..."
        ./nuctl create project cvat --platform local
    fi
}

function deploy(){
    echo "Deploying nuclio..."
    ./nuctl deploy --project-name cvat --path . --volume `pwd`/../../src:/opt/nuclio/src --platform local
}


while [[ $# -gt 0 ]]; do
    case $1 in 
    --checkpoint_path)
        checkpoint_path=$2
        ;;
    esac
    shift
done

download_checkpoint
download_nuclio
create_cvat_project
deploy
