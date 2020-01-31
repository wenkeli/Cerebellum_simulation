/****************************************************************************
** Meta object code from reading C++ file 'simthread.h'
**
** Created: Mon Aug 8 12:06:53 2011
**      by: The Qt Meta Object Compiler version 62 (Qt 4.7.2)
**
** WARNING! All changes made in this file will be lost!
*****************************************************************************/

#include "simthread.h"
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
       8,   14, // methods
       0,    0, // properties
       0,    0, // enums/sets
       0,    0, // constructors
       0,       // flags
       8,       // signalCount

 // signals: signature, parameters, type, tag, flags
      13,   11,   10,   10, 0x05,
      47,   44,   10,   10, 0x05,
      90,   11,   10,   10, 0x05,
     123,   11,   10,   10, 0x05,
     156,   11,   10,   10, 0x05,
     180,   10,   10,   10, 0x05,
     204,   10,   10,   10, 0x05,
     222,   11,   10,   10, 0x05,

       0        // eod
};

static const char qt_meta_stringdata_SimThread[] = {
    "SimThread\0\0,\0updateRaster(vector<bool>,int)\0"
    ",,\0updatePSH(vector<unsigned short>,int,bool)\0"
    "updateSCBCPCActs(SCBCPCActs,int)\0"
    "updateIONCPCActs(IONCPCActs,int)\0"
    "updateTotalAct(int,int)\0updateCSBackground(int)\0"
    "updateBlankDisp()\0"
    "updateActW(vector<bool>,vector<bool>)\0"
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
        case 0: updateRaster((*reinterpret_cast< vector<bool>(*)>(_a[1])),(*reinterpret_cast< int(*)>(_a[2]))); break;
        case 1: updatePSH((*reinterpret_cast< vector<unsigned short>(*)>(_a[1])),(*reinterpret_cast< int(*)>(_a[2])),(*reinterpret_cast< bool(*)>(_a[3]))); break;
        case 2: updateSCBCPCActs((*reinterpret_cast< SCBCPCActs(*)>(_a[1])),(*reinterpret_cast< int(*)>(_a[2]))); break;
        case 3: updateIONCPCActs((*reinterpret_cast< IONCPCActs(*)>(_a[1])),(*reinterpret_cast< int(*)>(_a[2]))); break;
        case 4: updateTotalAct((*reinterpret_cast< int(*)>(_a[1])),(*reinterpret_cast< int(*)>(_a[2]))); break;
        case 5: updateCSBackground((*reinterpret_cast< int(*)>(_a[1]))); break;
        case 6: updateBlankDisp(); break;
        case 7: updateActW((*reinterpret_cast< vector<bool>(*)>(_a[1])),(*reinterpret_cast< vector<bool>(*)>(_a[2]))); break;
        default: ;
        }
        _id -= 8;
    }
    return _id;
}

// SIGNAL 0
void SimThread::updateRaster(vector<bool> _t1, int _t2)
{
    void *_a[] = { 0, const_cast<void*>(reinterpret_cast<const void*>(&_t1)), const_cast<void*>(reinterpret_cast<const void*>(&_t2)) };
    QMetaObject::activate(this, &staticMetaObject, 0, _a);
}

// SIGNAL 1
void SimThread::updatePSH(vector<unsigned short> _t1, int _t2, bool _t3)
{
    void *_a[] = { 0, const_cast<void*>(reinterpret_cast<const void*>(&_t1)), const_cast<void*>(reinterpret_cast<const void*>(&_t2)), const_cast<void*>(reinterpret_cast<const void*>(&_t3)) };
    QMetaObject::activate(this, &staticMetaObject, 1, _a);
}

// SIGNAL 2
void SimThread::updateSCBCPCActs(SCBCPCActs _t1, int _t2)
{
    void *_a[] = { 0, const_cast<void*>(reinterpret_cast<const void*>(&_t1)), const_cast<void*>(reinterpret_cast<const void*>(&_t2)) };
    QMetaObject::activate(this, &staticMetaObject, 2, _a);
}

// SIGNAL 3
void SimThread::updateIONCPCActs(IONCPCActs _t1, int _t2)
{
    void *_a[] = { 0, const_cast<void*>(reinterpret_cast<const void*>(&_t1)), const_cast<void*>(reinterpret_cast<const void*>(&_t2)) };
    QMetaObject::activate(this, &staticMetaObject, 3, _a);
}

// SIGNAL 4
void SimThread::updateTotalAct(int _t1, int _t2)
{
    void *_a[] = { 0, const_cast<void*>(reinterpret_cast<const void*>(&_t1)), const_cast<void*>(reinterpret_cast<const void*>(&_t2)) };
    QMetaObject::activate(this, &staticMetaObject, 4, _a);
}

// SIGNAL 5
void SimThread::updateCSBackground(int _t1)
{
    void *_a[] = { 0, const_cast<void*>(reinterpret_cast<const void*>(&_t1)) };
    QMetaObject::activate(this, &staticMetaObject, 5, _a);
}

// SIGNAL 6
void SimThread::updateBlankDisp()
{
    QMetaObject::activate(this, &staticMetaObject, 6, 0);
}

// SIGNAL 7
void SimThread::updateActW(vector<bool> _t1, vector<bool> _t2)
{
    void *_a[] = { 0, const_cast<void*>(reinterpret_cast<const void*>(&_t1)), const_cast<void*>(reinterpret_cast<const void*>(&_t2)) };
    QMetaObject::activate(this, &staticMetaObject, 7, _a);
}
QT_END_MOC_NAMESPACE
