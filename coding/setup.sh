#!/bin/bash
apt-get update
apt-get install vim -y
apt-get install screen -y
apt-get install htop -y
apt-get install rsync -y
if [[ $PUBLIC_KEY ]]
then
    mkdir -p ~/.ssh
    chmod 700 ~/.ssh
    cd ~/.ssh
    echo $PUBLIC_KEY >> authorized_keys
    chmod 700 -R ~/.ssh
    service ssh start
fi
cd /workspace/unitartucs-thesis-tokenMerge/coding
pip3 install --no-input -r requirements.txt
python -m spacy download en_core_web_md
