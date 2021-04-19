#!/bin/bash


function download_from_drive() {
    fileid=${1};
    directory=${2};
    mkdir -p $directory;
    cd $directory;
    echo "Downloading $directory";
    FILE="${directory}.zip";
    if [[ -f "$FILE" ]]; then
        echo "$FILE exists, skipping download"
    else
        curl -c ./cookie -s -L "https://drive.google.com/uc?export=download&id=${fileid}" > /dev/null
        curl -Lb ./cookie "https://drive.google.com/uc?export=download&confirm=`awk '/download/ {print $NF}' \
        ./cookie`&id=${fileid}" -o ${FILE}
        rm ./cookie
    fi
    unzip ${FILE} && rm ${FILE};
    cd ..;
}

function download_models() {
    fileid="1_NXItk_1n3IqbiucVYRW85YixPfr21Vq";
    download_from_drive $fileid "models";
}

function download_data() {
    fileid="1P73uEOZ9As7uFUwBamHglxFwTZxj95Aa";
    download_from_drive $fileid "data";
}

sudo pip install -r requirements.txt;

download_data;
download_models;
