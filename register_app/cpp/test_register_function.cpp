#include "internal_include/test_register_function.h"

namespace test{
namespace test{


Object::Object(){
    std::cout << "Object constructor is called" << std::endl;
}

Object::~Object(){
    std::cout << "Object destructor is called" << std::endl;
}

Foo::Foo()
:Object()
{        
    std::cout << "Foo::constructor is called" <<std::endl;
}

Foo::Foo(const int& value)
:Object(),
m_value(value){}

void Foo::setValue(const int& value) {
    this->m_value = value;
}

int Foo::getValue() {
    return m_value;
}
Foo::~Foo(){
    std::cout << "Foo destructor is called" << std::endl;
}

Foo1::Foo1()
:Object()
{        
    std::cout << "Foo1::constructor is called" <<std::endl;
}

Foo1::Foo1(const int& value)
:Object(),
m_value(value){}

void Foo1::setValue(const int& value) {
    this->m_value = value;
}

int Foo1::getValue() {
    return m_value;
}
Foo1::~Foo1(){
    std::cout << "Foo1 destructor is called" << std::endl;
}

REGISTER(Foo);
REGISTER(Foo1);
} //test
} //test
