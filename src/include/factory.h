#ifndef _TEST_FACTORT
#define _TEST_FACTORT
#include <map>
#include <iostream>
#include <memory>

#include "utils.h"
#include "common.h"

namespace test{
namespace factory{
typedef void* (*func_ptr)();
typedef std::map<std::string, func_ptr> register_map;

class Factory
{
public:
    static func_ptr get_instance(const std::string instance_name);
    static void register_instance(const std::string instance_name, func_ptr instance);
};
} // namespace test
} // namespace factory
#endif //_TEST_FACTORT