/****************************************************************************
** Meta object code from reading C++ file 'mainw.h'
**
** Created: Fri Mar 18 20:22:40 2011
**      by: The Qt Meta Object Compiler version 62 (Qt 4.6.3)
**
** WARNING! All changes made in this file will be lost!
*****************************************************************************/

#include "mainw.h"
#if !defined(Q_MOC_OUTPUT_REVISION)
#error "The header file 'mainw.h' doesn't include <QObject>."
#elif Q_MOC_OUTPUT_REVISION != 62
#error "This file was generated using the moc from 4.6.3. It"
#error "cannot be used with the include files from this version of Qt."
#error "(The moc has changed too much.)"
#endif

QT_BEGIN_MOC_NAMESPACE
static const uint qt_meta_data_MainW[] = {

 // content:
       4,       // revision
       0,       // classname
       0,    0, // classinfo
       4,   14, // methods
       0,    0, // properties
       0,    0, // enums/sets
       0,    0, // constructors
       0,       // flags
       0,       // signalCount

 // slots: signature, parameters, type, tag, flags
       7,    6,    6,    6, 0x0a,
      24,    6,    6,    6, 0x0a,
      39,    6,    6,    6, 0x0a,
      53,    6,    6,    6, 0x0a,

       0        // eod
};

static const char qt_meta_stringdata_MainW[] = {
    "MainW\0\0dispSingleCell()\0dispAllCells()\0"
    "loadPSHFile()\0calcTempMetrics()\0"
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
        case 0: dispSingleCell(); break;
        case 1: dispAllCells(); break;
        case 2: loadPSHFile(); break;
        case 3: calcTempMetrics(); break;
        default: ;
        }
        _id -= 4;
    }
    return _id;
}
QT_END_MOC_NAMESPACE
