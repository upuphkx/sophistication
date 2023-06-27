#ifndef __TEST_SRC_TEST_H
#define __TEST_SRC_TEST_H
#include <malloc.h>
#include <gtest/gtest.h>

#include "register_app.h"
using namespace test::factory;

void testBitOp();
void testVectorAdd();
void testCUFFT();
void testMatMul();

#endif //__TEST_SRC_TEST_H