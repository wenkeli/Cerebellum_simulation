/****************************************************************************
** Meta object code from reading C++ file 'simctrlp.h'
**
** Created: Mon Aug 8 12:06:53 2011
**      by: The Qt Meta Object Compiler version 62 (Qt 4.7.2)
**
** WARNING! All changes made in this file will be lost!
*****************************************************************************/

#include "simctrlp.h"
#if !defined(Q_MOC_OUTPUT_REVISION)
#error "The header file 'simctrlp.h' doesn't include <QObject>."
#elif Q_MOC_OUTPUT_REVISION != 62
#error "This file was generated using the moc from 4.7.2. It"
#error "cannot be used with the include files from this version of Qt."
#error "(The moc has changed too much.)"
#endif

QT_BEGIN_MOC_NAMESPACE
static const uint qt_meta_data_SimCtrlP[] = {

 // content:
       5,       // revision
       0,       // classname
       0,    0, // classinfo
      12,   14, // methods
       0,    0, // properties
       0,    0, // enums/sets
       0,    0, // constructors
       0,       // flags
       0,       // signalCount

 // slots: signature, parameters, type, tag, flags
      10,    9,    9,    9, 0x0a,
      21,    9,    9,    9, 0x0a,
      32,    9,    9,    9, 0x0a,
      42,    9,    9,    9, 0x0a,
      59,    9,    9,    9, 0x0a,
      71,    9,    9,    9, 0x0a,
      92,   83,    9,    9, 0x0a,
     120,  112,    9,    9, 0x0a,
     147,  139,    9,    9, 0x0a,
     177,  166,    9,    9, 0x0a,
     207,  199,    9,    9, 0x0a,
     235,  229,    9,    9, 0x0a,

       0        // eod
};

static const char qt_meta_stringdata_SimCtrlP[] = {
    "SimCtrlP\0\0startSim()\0pauseSim()\0"
    "stopSim()\0dispSpikeRates()\0exportPSH()\0"
    "exportSim()\0dispMode\0changeDispMode(int)\0"
    "actMode\0changeActMode(int)\0pshMode\0"
    "changePSHMode(int)\0rasterMode\0"
    "changeRasterMode(int)\0srhMode\0"
    "changeSRHistMode(int)\0mzNum\0"
    "changeMZDispNum(int)\0"
};

const QMetaObject SimCtrlP::staticMetaObject = {
    { &QWidget::staticMetaObject, qt_meta_stringdata_SimCtrlP,
      qt_meta_data_SimCtrlP, 0 }
};

#ifdef Q_NO_DATA_RELOCATION
const QMetaObject &SimCtrlP::getStaticMetaObject() { return staticMetaObject; }
#endif //Q_NO_DATA_RELOCATION

const QMetaObject *SimCtrlP::metaObject() const
{
    return QObject::d_ptr->metaObject ? QObject::d_ptr->metaObject : &staticMetaObject;
}

void *SimCtrlP::qt_metacast(const char *_clname)
{
    if (!_clname) return 0;
    if (!strcmp(_clname, qt_meta_stringdata_SimCtrlP))
        return static_cast<void*>(const_cast< SimCtrlP*>(this));
    return QWidget::qt_metacast(_clname);
}

int SimCtrlP::qt_metacall(QMetaObject::Call _c, int _id, void **_a)
{
    _id = QWidget::qt_metacall(_c, _id, _a);
    if (_id < 0)
        return _id;
    if (_c == QMetaObject::InvokeMetaMethod) {
        switch (_id) {
        case 0: startSim(); break;
        case 1: pauseSim(); break;
        case 2: stopSim(); break;
        case 3: dispSpikeRates(); break;
        case 4: exportPSH(); break;
        case 5: exportSim(); break;
        case 6: changeDispMode((*reinterpret_cast< int(*)>(_a[1]))); break;
        case 7: changeActMode((*reinterpret_cast< int(*)>(_a[1]))); break;
        case 8: changePSHMode((*reinterpret_cast< int(*)>(_a[1]))); break;
        case 9: changeRasterMode((*reinterpret_cast< int(*)>(_a[1]))); break;
        case 10: changeSRHistMode((*reinterpret_cast< int(*)>(_a[1]))); break;
        case 11: changeMZDispNum((*reinterpret_cast< int(*)>(_a[1]))); break;
        default: ;
        }
        _id -= 12;
    }
    return _id;
}
QT_END_MOC_NAMESPACE
