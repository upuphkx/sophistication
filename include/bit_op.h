#ifndef __REGISTER_APP_BIT_OP_H
#define __REGISTER_APP_BIT_OP_H
#include <iostream>
#include <string>
#include <vector>
#include <memory>

#include "register.h"

namespace test{
namespace bit_op{

enum BITOPERATERSTATUS : uint32_t{
    kERROR,
    kSUCCESS,
    kUNSUPPORTTYPE
};

template<typename T>
void LogMessage(const T& message){
    std::cout << message << std::endl;
}

class OperatorBase{
public:
    virtual BITOPERATERSTATUS operator()(std::vector<int32_t>) = 0;
    virtual ~OperatorBase();
};

class BitWiseAndOp : public OperatorBase{
public: 
    int a = 1;
    BITOPERATERSTATUS BitWiseAndOpImpl(std::vector<int32_t>& input);

    virtual BITOPERATERSTATUS operator()(std::vector<int32_t> input) override final;
    virtual ~BitWiseAndOp();
};

class BitWiseOrOp : public OperatorBase{
public: 
    BITOPERATERSTATUS BitWiseOrOpImpl(std::vector<int32_t>& input);
    virtual BITOPERATERSTATUS operator()(std::vector<int32_t> input) override final;
    virtual ~BitWiseOrOp();
};

class BitWiseXOROp : public OperatorBase{
public: 
    BITOPERATERSTATUS BitWiseXOROpImpl(std::vector<int32_t>& input);
    virtual BITOPERATERSTATUS operator()(std::vector<int32_t> input) override final;
    virtual ~BitWiseXOROp();
};

class BitWiseNOTOp : public OperatorBase{
public: 
    BITOPERATERSTATUS BitWiseNOTOpImpl(std::vector<int32_t>& input);
    virtual BITOPERATERSTATUS operator()(std::vector<int32_t> input) override final;
    virtual ~BitWiseNOTOp();
};

class BitOpManager
{
public:
    std::vector<OperatorBase*> BasePtrVector_;
    void BasePtrVectorInit();
    BitOpManager();
    ~BitOpManager();
};

} // namespace bit_op
} // namespace test
#endif //__REGISTER_APP_BIT_OP_H