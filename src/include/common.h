#ifndef __TEST_COMMON_H
#define __TEST_COMMON_H
#include <vector>

// namespace tensor{
#define CHECK_RETURN_STATUS(expr, CURRENT_CALL_API)                                   \
        do {                                                                          \
        auto status = expr;                                                           \
        if(status == 1) {                                                             \
                std::cout << CURRENT_CALL_API << " exec success" << std::endl;        \
        }else if(status == 2){                                                        \
                 std::cout << CURRENT_CALL_API                                        \
                           << " not implement, return status(2)" << std::endl;        \
        }else{                                                                        \
                std::cout << CURRENT_CALL_API << " exec failed,"                      \
                          << " which in: "<< __FILE__                                 \
                          << ": " << __LINE__                                         \
                          << ", " << #expr << " return " << status                    \
                          << std::endl;                                               \
            }                                                                         \
        } while(0)

#define SAFE_DELETE_PTR(ptr)                                                            \
        do{                                                                             \
            if (ptr!=NULL)                                                              \
                {                                                                       \
                    delete ptr;                                                         \
                }                                                                       \
        }while(0)


typedef unsigned char       uint8_t;
typedef int                 int32_t;
typedef unsigned int        uint32_t;
typedef float               float32_t;
typedef double              float64_t;

enum DataType {
    kInt8_t,
    kInt16_t,
    kInt32_t,
    kInt64_t,
    kUInt8_t,
    kUInt16_t,
    kUInt32_t,
    kUInt64_t,
    kFloat32_t,
    kFloat64_t
};

class Log{
public:
    template<typename T>
    static void LogMessage(const T& message){
        std::cout << message << std::endl;
    }
};

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