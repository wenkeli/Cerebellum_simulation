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
    SVN=`svn up`
    if [[ $SVN == Updated* ]] || [[ $2 == "f" ]]
    then
        echo "Pulled an update or forced. Running Make."
        if [ -f $makefile ]
        then
            make -f $makefile cleanall
            make -f $makefile
        else
            make -f $backup cleanall
            make -f $backup 
        fi
    else
        echo "Already up to date"
    fi

    cd -
}

echo "Usage: $0 [f (force)]: Updates and builds. \"f\" flag will build regardless of update."

if [[ $1 == "f" ]]
then
    rm -f ../libs/*
fi

runupdate ../CXX_TOOLS_LIB/ $1
runupdate ../CBM_TOOLS_LIB/ $1
runupdate ../CBM_CORE_LIB/ $1
runupdate ../CBM_DATA_LIB/ $1
runupdate ../CBM_VISUAL_LIB/ $1
runupdate ../CBM_STATE_LIB/ $1
cd ../CBM_Params/ && svn up && cd -