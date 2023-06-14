#include "register.h"

// namespace test{
// namespace register{

Register::Register(const std::string& instance_name, const test::factory::func_ptr& instance)
{
    test::factory::Factory::register_instance(instance_name, instance);
}

// } // namespace register
// } // namespace test