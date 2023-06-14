# this project is how to use cpp register
g++ -fPIC -shared -o libregister.so factory/factory.cpp register/register.cpp test/test.cpp
g++ -o main main.cpp -L . -l register
