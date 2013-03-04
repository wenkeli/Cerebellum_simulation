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
    echo "Updating $1"
    cd $1
    SVN=`svn up`
    echo $SVN
    if [[ $SVN == Updated* ]] || [[ $2 == "f" ]]
    then
        if [ -f $makefile ]
        then
            make -f $makefile cleanall
            make -j2 -f $makefile
        else
            make -f $backup cleanall
            make -j2 -f $backup 
        fi
        ls lib/*.so
    fi

    cd - > /dev/null
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
echo "Updating ../CBM_Params/"
cd ../CBM_Params/ && svn up && cd -
