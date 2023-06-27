#ifndef _TSET_TEST_TEST
#define _TSET_TEST_TEST
#include "register.h"

namespace test{
namespace test{

class Object
{
public:
    Object();
    virtual void setValue(const int& value) = 0;
    virtual int getValue() = 0;
    virtual ~Object();
};

class Foo : public Object
{
private:
    int m_value;
public:
    Foo();

    Foo(const int& value);

    virtual void setValue(const int& value) override final;

    virtual int getValue() override final;
    virtual ~Foo();
};

class Foo1 : public Object
{
private:
    int m_value;
public:
    Foo1();

    Foo1(const int& value);

    virtual void setValue(const int& value) override final;

    virtual int getValue() override final;
    virtual ~Foo1();
};

} //test
} //test
#endif //_TSET_TEST_TEST