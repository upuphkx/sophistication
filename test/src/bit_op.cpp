#include "test.h"
using namespace test::factory;

void testBitOp(){
    Log::LogMessage("[TEST BIT OP BEGINING]");
    test::bit_op::BitOpManager* bit_op_manager = static_cast<test::bit_op::BitOpManager*>(test::factory::Factory::get_instance("BitOpManager")());
    std::vector<int32_t> input{127, 128, 255};
    for (auto ptr : bit_op_manager->BasePtrVector_)
    {
        CHECK_RETURN_STATUS((*ptr)(input), "bit_op test");
    }
    delete bit_op_manager;
    Log::LogMessage("[TEST BIT OP ENDING]");
}