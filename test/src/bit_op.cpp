#include "test.h"
using namespace test::factory;
using namespace test::bit_op;

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

TEST(SIMPLE_TEST, BitOp)
{

    test::bit_op::BitOpManager* bit_op_manager = static_cast<test::bit_op::BitOpManager*>(test::factory::Factory::get_instance("BitOpManager")());
    std::vector<int32_t> input{127, 128, 255};
    for (auto ptr : bit_op_manager->BasePtrVector_)
    {
        auto error = (*ptr)(input);
        EXPECT_EQ(error, 0);
        break;
    }
    delete bit_op_manager;

}

// #define TEST_MACRO(a, b, c)\
// TEST(SIMPLE_TEST, A##a##B##b##C##c)\
// {\

//     test::bit_op::BitOpManager* bit_op_manager = static_cast<test::bit_op::BitOpManager*>(test::factory::Factory::get_instance("BitOpManager")());\
//     std::vector<int32_t> input{127, 128, 255};\
//     for (auto ptr : bit_op_manager->BasePtrVector_)\
//     {\
//         auto error = (*ptr)(input);\
//         EXPECT_EQ(error, 0);\
//         break;\
//     }\
//     delete bit_op_manager;\

// }