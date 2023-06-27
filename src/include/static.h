#ifndef _STATIC_H
#define _STATIC_H

class Static
{
public:
    template<class T>
    static T& get_static_obj(){ //& is must
        static T static_obj;
        return static_obj;
    }
};

#endif //_STATIC_H