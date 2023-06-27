#include "internal_include/bit_op.h"

namespace test{
namespace bit_op{


OperatorBase::~OperatorBase(){}

BITOPERATERSTATUS BitWiseAndOp::BitWiseAndOpImpl(std::vector<int32_t>& input){
    if ((int32_t)(input[0] & input[1]) == input[2]){
        return kSUCCESS;
    } else {
        return kERROR;
    }
}

BITOPERATERSTATUS BitWiseAndOp::operator()(std::vector<int32_t> input) {
    BitWiseAndOpImpl(input);
}

 BitWiseAndOp::~BitWiseAndOp(){}

BITOPERATERSTATUS BitWiseOrOp::BitWiseOrOpImpl(std::vector<int32_t>& input){
    if ((int32_t)(input[0] | input[1]) == input[2]){
        return kSUCCESS;
    } else {
        return kERROR;
}
}

BITOPERATERSTATUS BitWiseOrOp::operator()(std::vector<int32_t> input)  {
    BitWiseOrOpImpl(input);
}

BitWiseOrOp::~BitWiseOrOp(){}

BITOPERATERSTATUS BitWiseXOROp::BitWiseXOROpImpl(std::vector<int32_t>& input){
    if ((int32_t)(input[0] ^ input[1]) == input[2]){
        return kSUCCESS;
    } else {
        return kERROR;
    }
}

BITOPERATERSTATUS BitWiseXOROp::operator()(std::vector<int32_t> input)  {
    BitWiseXOROpImpl(input);
}

 BitWiseXOROp::~BitWiseXOROp(){}

BITOPERATERSTATUS BitWiseNOTOp::BitWiseNOTOpImpl(std::vector<int32_t>& input){
    if ((int32_t)(~input[0]) == input[1]){
        return kSUCCESS;
    } else {
        return kERROR;
    }
}

BITOPERATERSTATUS BitWiseNOTOp::operator()(std::vector<int32_t> input)  {
    BitWiseNOTOpImpl(input);
}

 BitWiseNOTOp::~BitWiseNOTOp(){}

void BitOpManager::BasePtrVectorInit(){
    BasePtrVector_.push_back(new BitWiseAndOp());
    BasePtrVector_.push_back(new BitWiseOrOp());
    BasePtrVector_.push_back(new BitWiseXOROp());
    BasePtrVector_.push_back(new BitWiseNOTOp());
}

BitOpManager::BitOpManager(){
    BasePtrVectorInit();
}

BitOpManager::~BitOpManager(){
    for (OperatorBase* ptr : BasePtrVector_){
        delete ptr;
    }
}


REGISTER(BitOpManager);
} // namespace bit_op
} // namespace test