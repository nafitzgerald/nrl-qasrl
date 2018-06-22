#!/bin/bash

ggID='1FvMpjTfumVaSfwTOdWbJfEYFgGSAs0CS'  
ggURL='https://drive.google.com/uc?export=download'  
model_name="qasrl_parser_elmo"
archive="${model_name}.tar.gz"

echo "Downloading ${archive}"

filename="$(curl -sc /tmp/gcokie "${ggURL}&id=${ggID}" | grep -o '="uc-name.*</span>' | sed 's/.*">//;s/<.a> .*//')"  
getcode="$(awk '/_warning_/ {print $NF}' /tmp/gcokie)"  
curl -Lb /tmp/gcokie "${ggURL}&confirm=${getcode}&id=${ggID}" -o "${archive}"

echo "Unpacking archive in data directory"
target_dir="../data/${model_name}"
mkdir "${target_dir}"
tar zxvf "${archive}" -C "${target_dir}"

echo "Done."

