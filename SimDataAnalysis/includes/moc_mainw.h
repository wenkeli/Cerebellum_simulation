/****************************************************************************
** Meta object code from reading C++ file 'mainw.h'
**
** Created: Fri May 11 08:33:29 2012
**      by: The Qt Meta Object Compiler version 62 (Qt 4.7.4)
**
** WARNING! All changes made in this file will be lost!
*****************************************************************************/

#include "mainw.h"
#if !defined(Q_MOC_OUTPUT_REVISION)
#error "The header file 'mainw.h' doesn't include <QObject>."
#elif Q_MOC_OUTPUT_REVISION != 62
#error "This file was generated using the moc from 4.7.4. It"
#error "cannot be used with the include files from this version of Qt."
#error "(The moc has changed too much.)"
#endif

QT_BEGIN_MOC_NAMESPACE
static const uint qt_meta_data_MainW[] = {

 // content:
       5,       // revision
       0,       // classname
       0,    0, // classinfo
      26,   14, // methods
       0,    0, // properties
       0,    0, // enums/sets
       0,    0, // constructors
       0,       // flags
       0,       // signalCount

 // slots: signature, parameters, type, tag, flags
       7,    6,    6,    6, 0x0a,
      26,    6,    6,    6, 0x0a,
      44,    6,    6,    6, 0x0a,
      70,    6,    6,    6, 0x0a,
      95,    6,    6,    6, 0x0a,
     121,    6,    6,    6, 0x0a,
     141,    6,    6,    6, 0x0a,
     155,    6,    6,    6, 0x0a,
     176,    6,    6,    6, 0x0a,
     197,    6,    6,    6, 0x0a,
     216,    6,    6,    6, 0x0a,
     234,    6,    6,    6, 0x0a,
     251,    6,    6,    6, 0x0a,
     270,    6,    6,    6, 0x0a,
     297,    6,    6,    6, 0x0a,
     312,    6,    6,    6, 0x0a,
     335,    6,    6,    6, 0x0a,
     362,    6,    6,    6, 0x0a,
     378,    6,    6,    6, 0x0a,
     398,    6,    6,    6, 0x0a,
     419,    6,    6,    6, 0x0a,
     443,    6,    6,    6, 0x0a,
     464,    6,    6,    6, 0x0a,
     488,    6,    6,    6, 0x0a,
     502,    6,    6,    6, 0x0a,
     514,    6,    6,    6, 0x0a,

       0        // eod
};

static const char qt_meta_stringdata_MainW[] = {
    "MainW\0\0dispSingleCellNP()\0dispMultiCellNP()\0"
    "updateSingleCellDisp(int)\0"
    "updateMultiCellDisp(int)\0"
    "updateMultiCellBound(int)\0updateCellType(int)\0"
    "loadPSHFile()\0calcPFPCPlasticity()\0"
    "exportPFPCPlastAct()\0showGRInMFGOPSHs()\0"
    "showGROutGOPSHs()\0calcSpikeRates()\0"
    "exportSpikeRates()\0updateClusterCellType(int)\0"
    "makeClusters()\0updateClusterDisp(int)\0"
    "updateClusterCellDisp(int)\0dispClusterNP()\0"
    "dispClusterCellNP()\0dispInNetSpatialNP()\0"
    "updateInNetSpatial(int)\0exportInNetBinData()\0"
    "generate3DClusterData()\0loadSimFile()\0"
    "exportSim()\0exportSinglePSH()\0"
};

const QMetaObject MainW::staticMetaObject = {
    { &QMainWindow::staticMetaObject, qt_meta_stringdata_MainW,
      qt_meta_data_MainW, 0 }
};

#ifdef Q_NO_DATA_RELOCATION
const QMetaObject &MainW::getStaticMetaObject() { return staticMetaObject; }
#endif //Q_NO_DATA_RELOCATION

const QMetaObject *MainW::metaObject() const
{
    return QObject::d_ptr->metaObject ? QObject::d_ptr->metaObject : &staticMetaObject;
}

void *MainW::qt_metacast(const char *_clname)
{
    if (!_clname) return 0;
    if (!strcmp(_clname, qt_meta_stringdata_MainW))
        return static_cast<void*>(const_cast< MainW*>(this));
    return QMainWindow::qt_metacast(_clname);
}

int MainW::qt_metacall(QMetaObject::Call _c, int _id, void **_a)
{
    _id = QMainWindow::qt_metacall(_c, _id, _a);
    if (_id < 0)
        return _id;
    if (_c == QMetaObject::InvokeMetaMethod) {
        switch (_id) {
        case 0: dispSingleCellNP(); break;
        case 1: dispMultiCellNP(); break;
        case 2: updateSingleCellDisp((*reinterpret_cast< int(*)>(_a[1]))); break;
        case 3: updateMultiCellDisp((*reinterpret_cast< int(*)>(_a[1]))); break;
        case 4: updateMultiCellBound((*reinterpret_cast< int(*)>(_a[1]))); break;
        case 5: updateCellType((*reinterpret_cast< int(*)>(_a[1]))); break;
        case 6: loadPSHFile(); break;
        case 7: calcPFPCPlasticity(); break;
        case 8: exportPFPCPlastAct(); break;
        case 9: showGRInMFGOPSHs(); break;
        case 10: showGROutGOPSHs(); break;
        case 11: calcSpikeRates(); break;
        case 12: exportSpikeRates(); break;
        case 13: updateClusterCellType((*reinterpret_cast< int(*)>(_a[1]))); break;
        case 14: makeClusters(); break;
        case 15: updateClusterDisp((*reinterpret_cast< int(*)>(_a[1]))); break;
        case 16: updateClusterCellDisp((*reinterpret_cast< int(*)>(_a[1]))); break;
        case 17: dispClusterNP(); break;
        case 18: dispClusterCellNP(); break;
        case 19: dispInNetSpatialNP(); break;
        case 20: updateInNetSpatial((*reinterpret_cast< int(*)>(_a[1]))); break;
        case 21: exportInNetBinData(); break;
        case 22: generate3DClusterData(); break;
        case 23: loadSimFile(); break;
        case 24: exportSim(); break;
        case 25: exportSinglePSH(); break;
        default: ;
        }
        _id -= 26;
    }
    return _id;
}
QT_END_MOC_NAMESPACE
