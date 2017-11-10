#!/bin/bash

URL_1="https://blockchain.info/block-height/"
URL_3="?format=json"

START=493900
END=493938
for ((i=$START;i<=$END;i++)); do
    file="${i}.json"
    if [ -e ${file} ]
    then
        echo "${file} already exists"
    else
        curl "${URL_1}${i}${URL_3}" > "${file}"
    fi
done
