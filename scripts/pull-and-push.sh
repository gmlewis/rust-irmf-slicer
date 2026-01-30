#!/bin/bash -ex
git checkout master
git pull origin master
git pull origin master --tags
git push irmf master
