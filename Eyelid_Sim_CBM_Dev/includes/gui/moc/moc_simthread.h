/****************************************************************************
** Meta object code from reading C++ file 'simthread.h'
**
** Created: Wed Mar 14 16:29:12 2012
**      by: The Qt Meta Object Compiler version 62 (Qt 4.7.2)
**
** WARNING! All changes made in this file will be lost!
*****************************************************************************/

#include "../simthread.h"
#if !defined(Q_MOC_OUTPUT_REVISION)
#error "The header file 'simthread.h' doesn't include <QObject>."
#elif Q_MOC_OUTPUT_REVISION != 62
#error "This file was generated using the moc from 4.7.2. It"
#error "cannot be used with the include files from this version of Qt."
#error "(The moc has changed too much.)"
#endif

QT_BEGIN_MOC_NAMESPACE
static const uint qt_meta_data_SimThread[] = {

 // content:
       5,       // revision
       0,       // classname
       0,    0, // classinfo
       1,   14, // methods
       0,    0, // properties
       0,    0, // enums/sets
       0,    0, // constructors
       0,       // flags
       1,       // signalCount

 // signals: signature, parameters, type, tag, flags
      26,   11,   10,   10, 0x05,

       0        // eod
};

static const char qt_meta_stringdata_SimThread[] = {
    "SimThread\0\0,cellT,refresh\0"
    "updateSpatialW(std::vector<bool>,int,bool)\0"
};

const QMetaObject SimThread::staticMetaObject = {
    { &QThread::staticMetaObject, qt_meta_stringdata_SimThread,
      qt_meta_data_SimThread, 0 }
};

#ifdef Q_NO_DATA_RELOCATION
const QMetaObject &SimThread::getStaticMetaObject() { return staticMetaObject; }
#endif //Q_NO_DATA_RELOCATION

const QMetaObject *SimThread::metaObject() const
{
    return QObject::d_ptr->metaObject ? QObject::d_ptr->metaObject : &staticMetaObject;
}

void *SimThread::qt_metacast(const char *_clname)
{
    if (!_clname) return 0;
    if (!strcmp(_clname, qt_meta_stringdata_SimThread))
        return static_cast<void*>(const_cast< SimThread*>(this));
    return QThread::qt_metacast(_clname);
}

int SimThread::qt_metacall(QMetaObject::Call _c, int _id, void **_a)
{
    _id = QThread::qt_metacall(_c, _id, _a);
    if (_id < 0)
        return _id;
    if (_c == QMetaObject::InvokeMetaMethod) {
        switch (_id) {
        case 0: updateSpatialW((*reinterpret_cast< std::vector<bool>(*)>(_a[1])),(*reinterpret_cast< int(*)>(_a[2])),(*reinterpret_cast< bool(*)>(_a[3]))); break;
        default: ;
        }
        _id -= 1;
    }
    return _id;
}

// SIGNAL 0
void SimThread::updateSpatialW(std::vector<bool> _t1, int _t2, bool _t3)
{
    void *_a[] = { 0, const_cast<void*>(reinterpret_cast<const void*>(&_t1)), const_cast<void*>(reinterpret_cast<const void*>(&_t2)), const_cast<void*>(reinterpret_cast<const void*>(&_t3)) };
    QMetaObject::activate(this, &staticMetaObject, 0, _a);
}
QT_END_MOC_NAMESPACE
