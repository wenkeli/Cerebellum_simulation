#!/bin/bash
set -e

# This toggles the analysis code. Requires R, Rcpp, RInside
BUILD_ANALYSIS=1

# Includes for the Robocup associated code
ROBOCUP_INC="/home/matthew/projects/3Dsim/agents/nao-agent /home/matthew/projects/3Dsim/agents/nao-agent/core_utwalk /usr/local/include/simspark"
ROBOCUP_LIB_PATH="-L/home/matthew/projects/3Dsim/agents/nao-agent -L/usr/local/lib/simspark/"
ROBOCUP_LIB="-lnao -lrcssnet3D"

if [[ `hostname` == *tacc* ]]
then
    ROBOCUP_INC="$HOME/agents/nao-agent $HOME/agents/nao-agent/core_utwalk $HOME/local/include/simspark"
    ROBOCUP_LIB_PATH="-L$HOME/agents/nao-agent -L$HOME/local/lib/simspark/"
    ROBOCUP_LIB="-lnao -lrcssnet3D"
    BUILD_ANALYSIS=0
fi

# Includes for CBM Libraries
CBM_INC="../CXX_TOOLS_LIB/ ../CBM_TOOLS_LIB ../CBM_STATE_LIB ../CBM_CORE_LIB ../CBM_VISUAL_LIB ../CBM_DATA_LIB"
CBM_LIB_PATH="-L../libs"
CBM_LIB="-lcbm_tools -lcbm_state -lcbm_core -lcbm_visual -lcbm_data -lcxx_tools"

# Includes for Boost Libraries
BOOST_INC="$TACC_BOOST_INC"
BOOST_LIB_PATH="-L$TACC_BOOST_LIB"
BOOST_LIB="-lboost_program_options -lboost_system -lboost_filesystem"

# Includes for QT
QT_INC="/opt/apps/qt/4.7.0/include /usr/local/Trolltech/Qt-4.7.2/include /usr/include/qt4"
QT_LIB_PATH="-L/opt/apps/qt/4.7.0/lib -L/usr/local/Trolltech/Qt-4.7.2/lib -L/usr/lib/qt4"
QT_LIB="-lQtGui -lQtCore"

if [[ BUILD_ANALYSIS -eq 1 ]]
then
# Includes for R
    R_INC=`R CMD config --cppflags | cut -c3-`
    R_LIB=`R CMD config --ldflags`
    RCPP_INC=`echo 'Rcpp:::CxxFlags()' | R --vanilla --slave | cut -c3-`
    RCPP_LIB=`echo 'Rcpp:::LdFlags()'  | R --vanilla --slave`
    RINSIDE_INC=`echo 'RInside:::CxxFlags()' | R --vanilla --slave | cut -c3-`
    RINSIDE_LIB=`echo 'RInside:::LdFlags()'  | R --vanilla --slave`
    DEFINES+="BUILD_ANALYSIS"
fi

INC_PATH="/usr/local/cuda/include/ $BOOST_INC $CBM_INC $ROBOCUP_INC $QT_INC $R_INC $RCPP_INC $RINSIDE_INC"
LIB_PATH="$BOOST_LIB_PATH $CBM_LIB_PATH $ROBOCUP_LIB_PATH $QT_LIB_PATH $R_LIB $RCPP_LIB $RINSIDE_LIB"
LIBS="$ROBOCUP_LIB $CBM_LIB $BOOST_LIB $QT_LIB"

qmake -project -t app INCLUDEPATH+="$INC_PATH" LIBS+="$LIB_PATH $LIBS" OBJECTS_DIR="objs" MOC_DIR="moc" DEFINES+="$DEFINES" #QMAKE_CXXFLAGS+="-O1 -g"
qmake 
make -j2