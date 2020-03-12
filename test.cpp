#include <iostream>

using namespace std;


long fib(long num);


int main(){
    int num;
    cout<< "Enter number in your fibo ---"<<endl;
    cin>> num;
    cout<< "The "<< num << "th Fibonacy number is "<<fib(num)<<endl;
    return 0;
}
long fib(long num){
    if (num ==0 || num ==1)
        return num; 
    else{
        return fib(num - 1) + fib(num - 2);
    }
}
