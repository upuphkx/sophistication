#ifndef _STATIC_H
#define _STATIC_H

typedef int                 int32_t;
typedef float               float32_t;
typedef double              float64_t;
typedef unsigned int        uint32_t;
typedef unsigned char       uint8_t;

#define SAFE_DELETE_PTR(ptr)                                                            \
        do{                                                                             \
            if (ptr!=NULL)                                                             \
                {                                                                       \
                    delete ptr;                                                         \
                }                                                                       \
        }while(0)
        
class Log{
public:
    template<typename T>
    static void LogMessage(const T& message){
        std::cout << message << std::endl;
    }
};

class Static
{
public:
    template<class T>
    static T& get_static_obj(){ //& is must
        static T static_obj;
        return static_obj;
    }
};


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


#endif //_STATIC_H