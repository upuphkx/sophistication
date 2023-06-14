#include "factory.h"

namespace test{
namespace factory{
func_ptr Factory::get_instance(const std::string instance_name){
    if (Static::get_static_obj<register_map>().find(instance_name) == Static::get_static_obj<register_map>().end())
    {
        std::cerr << "not found instance: " << instance_name << std::endl;
        return NULL;
    } else {
        // std::cout << "instance found" << std::endl;
        return Static::get_static_obj<register_map>()[instance_name]; //return func_ptr type which used to create obj
    }
}
void Factory::register_instance(const std::string instance_name, func_ptr instance){
    // std::cout << "register instance" << std::endl;
    Static::get_static_obj<register_map>().insert(std::make_pair(instance_name, instance));
}
} //namespace factory
} //namespace test