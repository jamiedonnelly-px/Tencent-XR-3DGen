#!/bin/bash
set -e

#####from DevCloud enable_internet_proxy.sh####
#/etc/profile用于login shell;/etc/bashrc用于non-login shell
config_files=("${HOME}/.bashrc")

for config_file in ${config_files[@]}
do
        sed -i '/^http_proxy=/d'  $config_file
        sed -i '/^https_proxy=/d'  $config_file
        sed -i '/^no_proxy=/d'  $config_file
        sed -i '/^export http_proxy/d'  $config_file

        echo >> $config_file
        echo "http_proxy=http://http_proxy" >> $config_file
        echo "https_proxy=http://https_proxy" >> $config_file
        echo "no_proxy="no_proxy"" >> $config_file

        echo "export http_proxy https_proxy no_proxy" >> $config_file

        source $config_file
done

echo "enable internet proxy success!"

