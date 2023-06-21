#ifndef __TEST_COMMON_H
#define __TEST_COMMON_H
#include <vector>

#include "utils.h"

// namespace tensor{

class Shape
{
public:
    Shape(std::vector<uint32_t> shape){
        this->shape_= shape;
    }

    std::vector<uint32_t>& getValue(){
        return this->shape_;
    }

    void setValue(std::vector<uint32_t> shape){
        this->shape_ = shape;
        this->dim_ = this->shape_.size();
    }

    uint32_t getDim(){
        return this->dim_;
    }

private:
    uint32_t dim_;
    std::vector<uint32_t> shape_;
};


class Tensor
{
public:
    Tensor();
    Tensor(Shape* shape, void* buffer)
                :tensor_shape_(shape),
                buffer_(buffer){}
    Tensor(Shape* shape)
                :tensor_shape_(shape){}

    Tensor* setShape(Shape* shape){
        this->tensor_shape_ = shape;
        return this;
    }

    Shape* getShape(){
        return this->tensor_shape_;
    }

    Tensor* setBuffer(void* buffer){
        this->buffer_ = buffer;    
        return this;
    }

    void*& getBuffer(){
        return this->buffer_;
    }

    uint32_t getELementNum(){
        uint32_t element_num = 1;
        for(auto i : this->getShape()->getValue()){
            element_num *= i;
        }
        return element_num;
    }

    Tensor* setDataType(DataType datatype){
        this->datatype_ = datatype;
        return this;

    }

    DataType getDataType(){
        return this->datatype_;
    }

    size_t getByte(){
        // switch(type){
        //     case kInt32_t:
        //         size_t size = sizeof(int32_t) * this->getELementNum();
        //     case kFloat32_t:
        //         size_t size =  sizeof(float) * this->getELementNum();
        //     //TODO:impl more type
        //     default:
        //         size = 0;
        // }
        size_t size =  sizeof(float) * this->getELementNum();
        return size;
    }

private:
    Shape* tensor_shape_;
    void* buffer_;
    DataType datatype_ = kFloat32_t; // TODO:enum + switch

};


// } // namespace tensor
#endif //__TEST_COMMON_H