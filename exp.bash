#! /usr/bin/bash

UUID=$(cat /proc/sys/kernel/random/uuid)
name="name_${UUID}"
echo $name
