---
title: 实习收获 | C++入门踩坑介绍
author: 钟欣然
date: 2020-12-24 17:36:00 +0800
categories: [实习收获, 字节西瓜]
math: true
mermaid: true
---



# 头文件(.h)与源文件(.cpp)

* c++支持“分别编译”，即一个程序所有的内容，可以分成不同的部分，分别放在不同的 .cpp 文件里。.cpp 文件里的东西是相对独立的，在编译时不需要与其他文件互通，只需要在编译成目标文件后再与其他的目标文件做一次链接就行了。比如，在文件 a.cpp 中定义了一个全局函数 "void a(){}"，而在文件 b.cpp 中需要调用这个函数，调用之前先声明一下这个函数 "voida();"即可，文件 a.cpp 和文件 b.cpp 并不需要相互知道对方的存在，而是可以分别地对它们进行编译，编译成目标文件之后再链接，整个程序就可以运行了。
* 源文件放函数的定义，头文件放函数的声明，在一个 .cpp 文件中include一个头文件，相当于把这个头文件的内容copy到这个 .cpp 文件中，即相当于在这个 .cpp 文件中声明这些函数变量



# 标识符

* `:: `是作用域符，是运算符中等级最高的，它分为三种：
  * global scope（全局作用域符），用法（::name)，如在程序中的某一处想调用全局变量/函数a，那么就写成::a；
  * class scope（类作用域符），用法(class::name)，如在程序中的某一处想调用class A中的成员变量a，那么就写成A::a；
  * namespace scope（命名空间作用域符），用法(namespace::name)，如在程序中的某一处想调用namespace std中的cout成员，你就写成std::cout，即这里的cout对象是命名空间std中的cout（即标准库里边的cout）

* `auto`用于两种情况：
  * 声明变量时根据初始化表达式自动推断该变量的类型
  * 声明函数时函数返回值的占位符

* `->`用于指向类中的成员，类似于`.`：
  
  * 例如对一个结构体
  
  ```c++
  struct MyStruct{
    int a
  };
  ```
  
  * 对于这个结构体的实体，访问其中的变量使用`.`
  
  ```
  MyStruct s;
  s.a = 1
  ```
  
  * 对于这个结构体的指针，访问其中的变量需采用如下两种方式中的一种
  
  ```
  MyStruct *s;
  s->a = 1
  (*s).a = 1
  ```
  
* `*`与`&`：

  * `&`是一元运算符，返回操作数的内存地址。例如，如果 `var` 是一个整型变量，则 `&var` 是它的地址。
  * `*`是间接寻址运算符 *，是`&`运算符的补充，返回操作数所指定地址的变量的值。

  ```c++
  #include <iostream>
   
  using namespace std;
   
  int main ()
  {
     int  var;
     int  *ptr;
     int  val;
  
     var = 3000;
  
     // 获取 var 的地址
     ptr = &var;
  
     // 获取 ptr 的值
     val = *ptr;
     cout << "Value of var :" << var << endl;
     cout << "Value of ptr :" << ptr << endl;
     cout << "Value of val :" << val << endl;
  
     return 0;
  }
  
  // Value of var :3000
  // Value of ptr :0xbff64494
  // Value of val :3000
  ```

  

  

