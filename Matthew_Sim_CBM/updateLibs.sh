#!/bin/bash
set -e

function runupdate ()
{
    echo "Updating $1"
    cd $1
    SVN=`svn up`
    echo $SVN
    if [[ $2 == "f" ]]
    then
        make distclean
    fi
    ./build.sh
    cd - > /dev/null
}

echo "Usage: $0 [f (force rebuild)]: Updates and builds. \"f\" flag will clean and rebuild regardless of update."

if [[ $1 == "f" ]]
then
    rm -f ../libs/*
fi

runupdate ../CXX_TOOLS_LIB/ $1
runupdate ../CBM_STATE_LIB/ $1
runupdate ../CBM_TOOLS_LIB/ $1
runupdate ../CBM_DATA_LIB/ $1
runupdate ../CBM_VISUAL_LIB/ $1
runupdate ../CBM_CORE_LIB/ $1
runupdate ./ $1

echo "Updating ../CBM_Params/"
cd ../CBM_Params/ && svn up && cd -
