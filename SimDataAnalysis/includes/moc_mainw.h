/****************************************************************************
** Meta object code from reading C++ file 'mainw.h'
**
** Created: Tue Jul 12 13:02:45 2011
**      by: The Qt Meta Object Compiler version 62 (Qt 4.7.2)
**
** WARNING! All changes made in this file will be lost!
*****************************************************************************/

#include "mainw.h"
#if !defined(Q_MOC_OUTPUT_REVISION)
#error "The header file 'mainw.h' doesn't include <QObject>."
#elif Q_MOC_OUTPUT_REVISION != 62
#error "This file was generated using the moc from 4.7.2. It"
#error "cannot be used with the include files from this version of Qt."
#error "(The moc has changed too much.)"
#endif

QT_BEGIN_MOC_NAMESPACE
static const uint qt_meta_data_MainW[] = {

 // content:
       5,       // revision
       0,       // classname
       0,    0, // classinfo
      11,   14, // methods
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
     190,    6,    6,    6, 0x0a,
     202,    6,    6,    6, 0x0a,

       0        // eod
};

static const char qt_meta_stringdata_MainW[] = {
    "MainW\0\0dispSingleCellNP()\0dispMultiCellNP()\0"
    "updateSingleCellDisp(int)\0"
    "updateMultiCellDisp(int)\0"
    "updateMultiCellBound(int)\0updateCellType(int)\0"
    "loadPSHFile()\0calcPFPCPlasticity()\0"
    "loadSimFile()\0exportSim()\0exportSinglePSH()\0"
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
        case 8: loadSimFile(); break;
        case 9: exportSim(); break;
        case 10: exportSinglePSH(); break;
        default: ;
        }
        _id -= 11;
    }
    return _id;
}
QT_END_MOC_NAMESPACE
