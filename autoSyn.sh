#!/bin/sh
git status  
git add *  
git commit -m 'add eval.py'
#git commit -m 'add some results from Server'
git pull --rebase origin master   #domnload data
git push origin main            #upload data
git stash pop
