#!/bin/bash
set -e

makefile="ubuntu.mk"
backup="lonestar.mk"

if [[ `hostname` == *tacc* ]]
then
    makefile="lonestar.mk"
    backup="ubuntu.mk"
fi

function runupdate ()
{
    echo "Building $1"
    cd $1
    svn up
    if [ -f $makefile ]
    then
        make -f $makefile
    else
        make -f $backup
    fi
    cd -
}

rm -f ../libs/*
runupdate ../CXX_TOOLS_LIB/
runupdate ../CBM_TOOLS_LIB/
runupdate ../CBM_CORE_LIB/
runupdate ../CBM_DATA_LIB/
runupdate ../CBM_VISUAL_LIB/
runupdate ../CBM_STATE_LIB/
cd ../CBM_Params/ && svn up && cd -