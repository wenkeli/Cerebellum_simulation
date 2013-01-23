#!/bin/bash

qmake -project
qmake INCLUDEPATH+="../CXX_TOOLS_LIB/ ../CBM_TOOLS_LIB ../CBM_STATE_LIB ../CBM_CORE_LIB ../CBM_VISUAL_LIB ../CBM_DATA_LIB /usr/local/cuda/include/" LIBS+="-L../libs -L/opt/apps/boost/1.45.0/lib -lboost_program_options -lcbm_tools -lcbm_state -lcbm_core -lcbm_visual -lcbm_data -lcxx_tools"
make