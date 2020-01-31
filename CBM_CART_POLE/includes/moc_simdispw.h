/****************************************************************************
** Meta object code from reading C++ file 'simdispw.h'
**
** Created: Fri Apr 8 14:04:28 2011
**      by: The Qt Meta Object Compiler version 62 (Qt 4.7.2)
**
** WARNING! All changes made in this file will be lost!
*****************************************************************************/

#include "simdispw.h"
#if !defined(Q_MOC_OUTPUT_REVISION)
#error "The header file 'simdispw.h' doesn't include <QObject>."
#elif Q_MOC_OUTPUT_REVISION != 62
#error "This file was generated using the moc from 4.7.2. It"
#error "cannot be used with the include files from this version of Qt."
#error "(The moc has changed too much.)"
#endif

QT_BEGIN_MOC_NAMESPACE
static const uint qt_meta_data_SimDispW[] = {

 // content:
       5,       // revision
       0,       // classname
       0,    0, // classinfo
       7,   14, // methods
       0,    0, // properties
       0,    0, // enums/sets
       0,    0, // constructors
       0,       // flags
       0,       // signalCount

 // slots: signature, parameters, type, tag, flags
      12,   10,    9,    9, 0x0a,
      44,   41,    9,    9, 0x0a,
      85,   10,    9,    9, 0x0a,
     116,   10,    9,    9, 0x0a,
     147,   10,    9,    9, 0x0a,
     169,    9,    9,    9, 0x0a,
     191,    9,    9,    9, 0x0a,

       0        // eod
};

static const char qt_meta_stringdata_SimDispW[] = {
    "SimDispW\0\0,\0drawRaster(vector<bool>,int)\0"
    ",,\0drawPSH(vector<unsigned short>,int,bool)\0"
    "drawSCBCPCActs(SCBCPCActs,int)\0"
    "drawIONCPCActs(IONCPCActs,int)\0"
    "drawTotalAct(int,int)\0drawCSBackground(int)\0"
    "drawBlankDisp()\0"
};

const QMetaObject SimDispW::staticMetaObject = {
    { &QWidget::staticMetaObject, qt_meta_stringdata_SimDispW,
      qt_meta_data_SimDispW, 0 }
};

#ifdef Q_NO_DATA_RELOCATION
const QMetaObject &SimDispW::getStaticMetaObject() { return staticMetaObject; }
#endif //Q_NO_DATA_RELOCATION

const QMetaObject *SimDispW::metaObject() const
{
    return QObject::d_ptr->metaObject ? QObject::d_ptr->metaObject : &staticMetaObject;
}

void *SimDispW::qt_metacast(const char *_clname)
{
    if (!_clname) return 0;
    if (!strcmp(_clname, qt_meta_stringdata_SimDispW))
        return static_cast<void*>(const_cast< SimDispW*>(this));
    return QWidget::qt_metacast(_clname);
}

int SimDispW::qt_metacall(QMetaObject::Call _c, int _id, void **_a)
{
    _id = QWidget::qt_metacall(_c, _id, _a);
    if (_id < 0)
        return _id;
    if (_c == QMetaObject::InvokeMetaMethod) {
        switch (_id) {
        case 0: drawRaster((*reinterpret_cast< vector<bool>(*)>(_a[1])),(*reinterpret_cast< int(*)>(_a[2]))); break;
        case 1: drawPSH((*reinterpret_cast< vector<unsigned short>(*)>(_a[1])),(*reinterpret_cast< int(*)>(_a[2])),(*reinterpret_cast< bool(*)>(_a[3]))); break;
        case 2: drawSCBCPCActs((*reinterpret_cast< SCBCPCActs(*)>(_a[1])),(*reinterpret_cast< int(*)>(_a[2]))); break;
        case 3: drawIONCPCActs((*reinterpret_cast< IONCPCActs(*)>(_a[1])),(*reinterpret_cast< int(*)>(_a[2]))); break;
        case 4: drawTotalAct((*reinterpret_cast< int(*)>(_a[1])),(*reinterpret_cast< int(*)>(_a[2]))); break;
        case 5: drawCSBackground((*reinterpret_cast< int(*)>(_a[1]))); break;
        case 6: drawBlankDisp(); break;
        default: ;
        }
        _id -= 7;
    }
    return _id;
}
QT_END_MOC_NAMESPACE
