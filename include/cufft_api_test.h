#ifndef __REGISTER_CUFFT_API_TEST_H
#define __REGISTER_CUFFT_API_TEST_H
#include <cufftXt.h>
#include <cuda_runtime.h>

#include <complex>
#include <iostream>
#include <random>
#include <vector>
#include <string>
#include <array>

#include "register.h"

typedef std::complex<float32_t> data_type;
using cpudata_t = std::vector<std::complex<float32_t>>;
using gpus_t = std::vector<int32_t>;
using dim_t = std::array<size_t, 3>;

namespace test{
namespace cufft_api{
//CUDA API ERROR CHECKING
#define CUDA_RT_CALL(call)                                                \
    do {                                                                  \
        auto status = static_cast<cudaError_t>(call);                     \
        if(status != cudaSuccess)                                         \
            fprintf(stderr,                                               \
                "ERROR: CUDA RT call \"%s\" in line %d of file %s failed "\
                "with "                                                   \
                "%s (%d).\n",                                             \
                #call,                                                    \
                __LINE__,                                                 \
                __FILE__,                                                 \
                cudaGetErrorString(status),                               \
                status);                                                  \
    } while(0)

//CUFFT API ERROR CHECKING
#define CUFFT_CALL(call)                                                  \
    do {                                                                  \
        auto status = static_cast<cufftResult>(call);                     \
        if(status != CUFFT_SUCCESS)                                       \
            fprintf(stderr,                                               \
                "ERROR: CUFFT call \"%s\" in line %d of file %s failed "  \
                "with "                                                   \
                "code (%d).\n",                                           \
                #call,                                                    \
                __LINE__,                                                 \
                __FILE__,                                                 \
                status);                                                  \
    } while(0)

enum TESTCUFFTAPIRETURNSTATUS : uint8_t {
    TEST_CUFFT_SUCCESS       = 1,
    TEST_CUFFT_ERROR         = 0,
    TEST_CUFFT_NOT_IMPLEMENT = 2,
    TETS_CUFFT_UNKONW_ERROR  = 3
};

class CufftAPITest{
public:
    virtual TESTCUFFTAPIRETURNSTATUS cufftAPIImpl(const uint32_t& n,
                                     const uint32_t& batch_size) = 0;
    virtual ~CufftAPITest(){}
};

class cufft1DC2CTest : public CufftAPITest{
public:
    virtual TESTCUFFTAPIRETURNSTATUS cufftAPIImpl(const uint32_t& n,
                                     const uint32_t& batch_size) override final;
    virtual ~cufft1DC2CTest(){}
};

class cufft1DR2CTest : public CufftAPITest{
public:
    virtual TESTCUFFTAPIRETURNSTATUS cufftAPIImpl(const uint32_t& n,
                                     const uint32_t& batch_size) override final;
    virtual ~cufft1DR2CTest(){}
};

class cufft1DMGPUC2CTest : public CufftAPITest{
public:
    virtual TESTCUFFTAPIRETURNSTATUS cufftAPIImpl(const uint32_t& n,
                                     const uint32_t& batch_size) override final;
    void fill_array(cpudata_t &array);
    void single(dim_t fft, int &batch_size, cpudata_t &h_data_in, cpudata_t &h_data_out);
    void spmg(dim_t fft, int &batch_size, gpus_t gpus, cpudata_t &h_data_in, cpudata_t &h_data_out,
          cufftXtSubFormat_t subformat);        
    virtual ~cufft1DMGPUC2CTest(){}
};

class cufft2DC2RTest : public CufftAPITest{
public:
    virtual TESTCUFFTAPIRETURNSTATUS cufftAPIImpl(const uint32_t& n,
                                     const uint32_t& batch_size) override final;
    virtual ~cufft2DC2RTest(){}
};

class cufft3DC2CTest : public CufftAPITest{
public:
    virtual TESTCUFFTAPIRETURNSTATUS cufftAPIImpl(const uint32_t& n,
                                     const uint32_t& batch_size) override final;
    virtual ~cufft3DC2CTest(){}
};

class cufft3DMGPUC2CTest : public CufftAPITest{
public:
    virtual TESTCUFFTAPIRETURNSTATUS cufftAPIImpl(const uint32_t& n,
                                     const uint32_t& batch_size) override final;
    virtual ~cufft3DMGPUC2CTest(){}
};

class cufft3DMGPUR2CC2RTest : public CufftAPITest{
public:
    virtual TESTCUFFTAPIRETURNSTATUS cufftAPIImpl(const uint32_t& n,
                                     const uint32_t& batch_size) override final;
    virtual ~cufft3DMGPUR2CC2RTest(){}
};

} // cufft_api
} // test
#endif // __REGISTER_CUFFT_API_TEST_H