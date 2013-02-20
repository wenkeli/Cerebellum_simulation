#!/bin/bash

# Includes for the Robocup associated code
ROBOCUP_INC="/home/matthew/projects/3Dsim/agents/nao-agent /home/matthew/projects/3Dsim/agents/nao-agent/core_utwalk /usr/local/include/simspark"
ROBOCUP_LIB_PATH="-L/home/matthew/projects/3Dsim/agents/nao-agent -L/usr/local/lib/simspark/"
ROBOCUP_LIB="-lnao -lrcssnet3D"

# Includes for CBM Libraries
CBM_INC="../CXX_TOOLS_LIB/ ../CBM_TOOLS_LIB ../CBM_STATE_LIB ../CBM_CORE_LIB ../CBM_VISUAL_LIB ../CBM_DATA_LIB"
CBM_LIB_PATH="-L../libs"
CBM_LIB="-lcbm_tools -lcbm_state -lcbm_core -lcbm_visual -lcbm_data -lcxx_tools"

# Includes for Boost Libraries
BOOST_INC=""
BOOST_LIB_PATH="-L/opt/apps/boost/1.45.0/lib"
BOOST_LIB="-lboost_program_options"

INC_PATH="/usr/local/cuda/include/ $CBM_INC $ROBOCUP_INC"
LIB_PATH="$BOOST_LIB_PATH $CBM_LIB_PATH $ROBOCUP_LIB_PATH"
LIBS="$ROBOCUP_LIB $CBM_LIB $BOOST_LIB"

qmake -project
qmake INCLUDEPATH+="$INC_PATH" LIBS+="$LIB_PATH $LIBS"
make 