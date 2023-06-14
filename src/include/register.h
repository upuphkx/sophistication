#ifndef _TEST_REGISTER_REGISTER
#define _TEST_REGISTER_REGISTER
#include "factory.h"
// namespace test {
// namespace register {


class Register
{
public:
    Register(const std::string& instance_name, const test::factory::func_ptr& instance);
};

// #define REGISTER(instance_name)                                                                                            \
//     class CreatInstance##instance_name                                                                                     \
//     {                                                                                                                      \
//     public:                                                                                                                \
//         static void* instance (){                                                                                          \
//             return new instance_name;                                                                                      \
//         }                                                                                                                  \
//     private:                                                                                                               \
//         static const Register _static_register;};                                                                          \
//     const Register CreatInstance##instance_name::_static_register(#instance_name, instance);                               \

#define REGISTER(instance_name)                                                                                               \
        void* instance_name##instance (){return new instance_name;}                                                           \
        const Register instance_name##_static_register(#instance_name, instance_name##instance);                              \

// } // namespace register
// } // namespace test
#endif //_TEST_REGISTER_REGISTER
