#!/usr/bin/env bash

NAME="Yan Errol"
echo "Hello, $NAME!"

echo $NAME
echo "$NAME"
echo "${NAME}!"
echo ${NAME/J/j}

echo ${NAME:0:2}
echo ${NAME::2}
echo ${NAME:(-1)}
echo ${NAME:(-5):1}
echo ${NAME:erle}
echo "///////////////////"
length=5
echo ${NAME:0:length}

STR = "/Users/yanerrol/Desktop/bash_shell_tutorial"
echo ${STR%bash_shell_tutorial}

get_name(){
echo "Yan Errol"

}
echo "You are $(get_name)"

echo "I'm in $(pwd)"
echo "I'm in $pwd "

if [[ -z "$string" ]]; then
  echo "String is empty"
elif [[ -n "$string" ]]; then
  echo "String is not empty"
fi

echo {A,B}.js

