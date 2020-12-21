---
title: 实习收获 | Go语言学习从零开始干货整理
author: 钟欣然
date: 2020-12-12 00:45:00 +0800
categories: [实习收获, 字节飞书]
math: true
mermaid: true
---

本文整理自[菜鸟教程](https://www.runoob.com/go/go-tutorial.html)

#  安装Go语言
安装包[下载地址](https://golang.org/dl/)，Windows系统默认安装即可

# 执行Go程序
示例程序步骤如下：

1. 建立txt文件，将文件拓展名.txt修改为.go（设置文件查看时显示拓展名，然后直接修改即可），用notepad++等编辑软件编写以下程序

```go
package main

import "fmt"

func main(){
    fmt.Println("Hello, World!")
}
```

2. 打开命令行（Win+R键，输入cmd回车唤出命令行），将路径调整为当前路径，示例如下：

```
# cd 拟修改路径进行转换
C:\Users\Admin>cd C:\Users\Admin\Desktop
# C盘切换至其它盘，需要先切换盘，再cd
C:\Users\Admin>D:
D:>cd D:\go语言学习
```

3. 执行程序，输出Hello World!即执行成功

```
D:\go语言学习>go run hello.go
```
4. 我们还可以使用 go build 命令来生成二进制文件，再直接执行该程序即可

```
D:\go语言学习>go bulid hello.go
D:\go语言学习>ls # 查看当前路径下的文件
D:\go语言学习>hello
```
# Go 语言结构
以示例程序为例：

```go
package main

import "fmt"

func main(){
    fmt.Println("Hello, World!")
}
```
- **包声明**：第一行代码package main定义了包名。你必须在源文件中非注释的第一行指明这个文件属于哪个包，如：package main。package main表示一个可独立执行的程序，每个 Go 应用程序都包含一个名为 main 的包
- **引入包**：下一行 import "fmt" 告诉 Go 编译器这个程序需要使用 fmt 包（的函数，或其他元素），fmt 包实现了格式化 IO（输入/输出）的函数
- **函数**：下一行 func main() 是程序开始执行的函数。main 函数是每一个可执行程序所必须包含的，一般来说都是在启动后第一个执行的函数（如果有 init() 函数则会先执行该函数）
- **变量**
- **语句 & 表达式**
- **注释**：单行注释//，多行注释被/* */圈住，多行注释一般用于包的文档描述或注释成块的代码片段

> 注意：
> - 当标识符（包括常量、变量、类型、函数名、结构字段等等）以一个大写字母开头，如：Group1，那么使用这种形式的标识符的对象就可以被外部包的代码所使用（客户端程序需要先导入这个包），这被称为导出（像面向对象语言中的 public）；标识符如果以小写字母开头，则对包外是不可见的，但是他们在整个包的内部是可见并且可用的（像面向对象语言中的 protected ）
> - fmt.Println(...) 可以将字符串输出到控制台，并在最后自动增加换行字符 \n。使用 fmt.Print("hello, world\n") 可以得到相同的结果。Print 和 Println 这两个函数也支持使用变量，如：fmt.Println(arr)。如果没有特别指定，它们会以默认的打印格式将变量 arr 输出到控制台。
> - { 不能单独放在一行的开头

# 基础语法
- **Go标记**：Go 程序可以由多个标记组成，可以是关键字，标识符，常量，字符串，符号。如以下Go语句由6个标记组成

```go
fmt.Println("Hello, World!")
```
6个标记分别为
```go
fmt
.
Println
(
"Hello, World!"
)
```

- **行分隔符**：一行代表一个语句结束，不需要以分号结尾，如果将多个语句写在一行，则必须使用分号人为区分
- **标识符**：标识符用来命名变量、类型等程序实体。一个标识符实际上就是一个或是多个字母(A~Z和a~z)数字(0~9)、下划线_组成的序列，但是第一个字符必须是字母或下划线而不能是数字
- **字符串连接**：可以使用加号实现，如

```go
fmt.Println("Google" + "Runoob")
```

- **关键字**：

下面列举了 Go 代码中会使用到的 25 个关键字或保留字：

![在这里插入图片描述](https://img-blog.csdnimg.cn/20200302143054272.png#pic_center)

除了以上介绍的这些关键字，Go 语言还有 36 个预定义标识符：

![在这里插入图片描述](https://img-blog.csdnimg.cn/20200302143111625.png#pic_center)

程序一般由关键字、常量、变量、运算符、类型和函数组成。

分隔符：括号 ()，中括号 [] 和大括号 {}。

标点符号：点. 逗号, 分号; 冒号: 省略号…
- **空格**：变量的声明必须使用空格隔开

```go
var age int
```
# 数据类型
- **布尔型**：true false
- **数字类型**：整形int和浮点型float32、float64
- **字符串类型**：Go 的字符串是由单个字节连接起来的，使用 UTF-8 编码标识 Unicode 文本
- **派生类型**：如指针类型（Pointer）、数组类型、结构化类型（struct）、Channel类型、函数类型、切片类型、接口类型、Map类型

详细介绍数字类型如下：

- Go也有基于架构的类型，例如：int、uint 和 uintptr
![在这里插入图片描述](https://img-blog.csdnimg.cn/20200302143008641.png#pic_center)
- 浮点型
![在这里插入图片描述](https://img-blog.csdnimg.cn/20200302143224322.png#pic_center)
- 其他数字类型：
![在这里插入图片描述](https://img-blog.csdnimg.cn/20200302143256408.png#pic_center)

# 变量
声明变量的一般形式：

```go
var identifier type
// 一次声明多个变量
var identifier1, identifier2 type
```
## 变量声明

- **指定变量类型**：可以给初始值，也可不给，不给时默认为零值

```go
var a string = "Runoob"
var b int
var c, d = int  // 多变量
```

有关零值的说明：数值类型为0，布尔类型为false，字符串为""，以下几种类型为nil

```go
var a *int
var a []int
var a map[string] int
var a chan int
var a func(string) int
var a error  // error 是接口
```
- **根据值自行判定变量类型**

```go
var e = 10
var f = true
var g, h = 10, 20  // 多变量
```
- **省略var，用:=**：只能被用在函数体内，而不可以用于全局变量的声明与赋值。注意:=左侧如果没有声明新的变量，就产生编译错误

```go
i := 3
j, k := 3, 4  // 多变量，左侧的变量不能是已经被声明过的，否则会报错
```
- **因式分解关键字**：一般用于声明全局变量

```go
var (
    l int
    m string
)
```

## 值类型和引用类型

所有像 int、float、bool 和 string 这些基本类型都属于值类型，使用这些类型的变量直接指向存在内存中的值，当使用等号 = 将一个变量的值赋值给另一个变量时，如：j = i，实际上是在内存中将 i 的值进行了拷贝。

![在这里插入图片描述](https://img-blog.csdnimg.cn/20200302145130780.png#pic_center)

你可以通过 &i 来获取变量 i 的内存地址，例如：0xf840000040（每次的地址都可能不一样）。值类型的变量的值存储在栈中。内存地址会根据机器的不同而有所不同，甚至相同的程序在不同的机器上执行后也会有不同的内存地址。因为每台机器可能有不同的存储器布局，并且位置分配也可能不同。

![在这里插入图片描述](https://img-blog.csdnimg.cn/20200302145111717.png#pic_center)

更复杂的数据通常会需要使用多个字，这些数据一般使用引用类型保存。一个引用类型的变量 r1 存储的是 r1 的值所在的内存地址（数字），或内存地址中第一个字所在的位置。这个内存地址为称之为指针，这个指针实际上也被存在另外的某一个字中。

同一个引用类型的指针指向的多个字可以是在连续的内存地址中（内存布局是连续的），这也是计算效率最高的一种存储形式；也可以将这些字分散存放在内存中，每个字都指示了下一个字所在的内存地址。

当使用赋值语句 r2 = r1 时，只有引用（地址）被复制。如果 r1 的值被改变了，那么这个值的所有引用都会指向被修改后的内容，在这个例子中，r2 也会受到影响。

![在这里插入图片描述](https://img-blog.csdnimg.cn/20200302145159281.png#pic_center)

> 注意：
> - **不能二次声明**：如果在相同的代码块中，我们不可以再次对于相同名称的变量使用初始化声明，例如：在声明a := 30后，再a := 20 就是不被允许的，编译器会提示错误 no new variables on left side of :=，但是 a = 20 是可以的，因为这是给相同的变量赋予一个新的值
> - **不能不声明就用**：如果你在定义变量 a 之前使用它，则会得到编译错误 undefined: a
> - **不能声明了不用**：如果你声明了一个局部变量却没有在相同的代码块中使用它，同样会得到编译错误，例如下面这个例子当中的变量 a，尝试编译这段代码将得到错误 a declared and not used。

```go
package main

import "fmt"

func main() {
   var a string = "abc"
   fmt.Println("hello, world")
}
```
> - **全局变量可以声明了不用**
> - **交换两个变量的值**：a, b = b, a
> - **空白标识符**：空白标识符 _ 也被用于抛弃值，如值 5 在：_, b = 5, 7 中被抛弃。_ 实际上是一个只写变量，你不能得到它的值。这样做是因为 Go 语言中你必须使用所有被声明的变量，但有时你并不需要使用从一个函数得到的所有返回值
> - **并行赋值可以用于函数返回多个值时**：比如这里的 val 和错误 err 是通过调用 Func1 函数同时得到：val, err = Func1(var1)

# 常量

常量是一个简单值的标识符，在程序运行时，不会被修改的量，数据类型只可以是布尔型、数字型（整数型、浮点型和复数）和字符串型。定义格式如下：

```go
const identifier [type] = value  // type可省略
const c_name1, c_name2 = value1, value2  // 多变量
const (  // 用作枚举
    Unknown = 0
    Female = 1
    Male = 2
)
```
常量可以用len(), cap(), unsafe.Sizeof()函数计算表达式的值。常量表达式中，函数必须是内置函数，否则编译不过：

```go
package main

import "unsafe"
const (
    a = "abc"
    b = len(a)
    c = unsafe.Sizeof(a)
)

func main(){
    println(a, b, c)
}
```
输出结果为：

	abc 3 16

## iota
iota，特殊常量，可以认为是一个可以被编译器修改的常量。iota 在const关键字出现时将被重置为0(const 内部的第一行之前)，const 中每新增一行常量声明将使 iota 计数一次(iota 可理解为 const 语句块中的行索引)。iota 可以被用作枚举值：

```go
const (
    a = iota
    b = iota
    c = iota
)
```
第一个 iota 等于 0，每当 iota 在新的一行被使用时，它的值都会自动加 1；所以 a=0, b=1, c=2 可以简写为如下形式：

```go
const (
    a = iota
    b
    c
)
```

用法示例：

```go
package main

import "fmt"

func main() {
    const (
            a = iota   //0
            b          //1
            c          //2
            d = "ha"   //独立值，iota += 1
            e          //"ha"   iota += 1
            f = 100    //iota +=1
            g          //100  iota +=1
            h = iota   //7,恢复计数
            i          //8
    )
    fmt.Println(a,b,c,d,e,f,g,h,i)
}
```

# 运算符

## 六类运算符
- 算术运算符：假定A为10，B为20

![在这里插入图片描述](https://img-blog.csdnimg.cn/20200302151947438.png#pic_center)

- 关系运算符：假定A为10，B为20

![在这里插入图片描述](https://img-blog.csdnimg.cn/20200302152141779.png#pic_center)

- 逻辑运算符：假定A为true，B为false

![在这里插入图片描述](https://img-blog.csdnimg.cn/2020030215230066.png#pic_center)

- 位运算符：位运算符对整数在内存中的二进制位进行操作，假定A=60，B=13

![在这里插入图片描述](https://img-blog.csdnimg.cn/20200302153041630.png#pic_center)

示例如下：
A = 0011 1100
B = 0000 1101
A&B = 0000 1100
A|B = 0011 1101
A^B = 0011 0001
A<<2 = 1111 0000
A>>2 = 0000 1111

- 赋值运算符

![在这里插入图片描述](https://img-blog.csdnimg.cn/20200302153245463.png#pic_center)

- 其他运算符：

![在这里插入图片描述](https://img-blog.csdnimg.cn/20200302153345264.png#pic_center)

## 运算符优先级
有些运算符拥有较高的优先级，二元运算符的运算方向均是从左至右。下表列出了所有运算符以及它们的优先级，由上至下代表优先级由高到低，可以通过使用括号来临时提升某个表达式的整体运算优先级

![在这里插入图片描述](https://img-blog.csdnimg.cn/20200302153636192.png#pic_center)

# 条件语句
Go 语言提供了以下几种条件判断语句：

![在这里插入图片描述](https://img-blog.csdnimg.cn/20200302154005662.png#pic_center)

##  基本语法
```
// if语句
if () {
    …
}
// if…else语句
if () {
    …
} else {
    …
}
// if嵌套语句
if () {
    if () {
        …
    }
} else {
    …
}
// switch语句
switch var1 {
    case val1:
        …
    case val2:  // 可以定义任意数量的 case
        …
    default:
        …
}
// Type switch语句用于判断变量类型
switch x.(type) {
    case type1:
        …
    case type2:  // 可以定义任意数量的 case
        …
    default:
        …
}
// select语句
select {
    case communication clause:
        …
    case communication clause:  // 可以定义任意数量的 case
        …
    default:
        …
}
```

## switch语句说明

- switch语句用于基于不同条件执行不同动作，每一个 case 分支都是唯一的，从上至下逐一测试，直到匹配为止。匹配项后不需要加break，匹配成功后不会执行其他 case
- 可以同时测试多个可能符合条件的值，使用逗号分割它们，例如：case val1, val2, val3
- 如果需要执行后面的 case，可以使用 fallthrough，使用 fallthrough 会强制执行后面的 case 语句，fallthrough 不会判断下一条 case 的表达式结果是否为 true



## select语句说明

- 每个case都必须是一个通道，每个通道表达式都会被求值
- 如果仅有一个case可以进行，则执行该case；如果有多个case都可以进行，则随机公平选出一个case执行；如果没有case可以进行，则有default执行default，没有default则select被阻塞，直到某个通信可以运行
- [具体示例](https://www.jianshu.com/p/2a1146dc42c3)

> 注意：
> - [ ] Go 没有三目运算符，所以不支持 ?: 形式的条件判断
> - [ ] 有关通道之后可以查看更多

```go
ch := make(chan int)  //初始化通道
x := 1
ch <- x  // 接收x的值
x <- ch  // 将通道中的值赋给x
```
# 循环语句

Go 语言提供了以下几种类型循环处理语句：

![在这里插入图片描述](https://img-blog.csdnimg.cn/2020030217234856.png#pic_center)

## 基本语法

- init： 一般为赋值表达式，给控制变量赋初值
- condition： 关系表达式或逻辑表达式，循环控制条件
- post： 一般为赋值表达式，给控制变量增量或减量

```
// for循环有四种形式
for init; condition; post {  // init和post可以省略，但对应的分号不能省略
    …
}
for condition {
    …
}
for {  // 无限循环，可以用ctrl+c停止无限循环
    …
}
for key, value := range oldmap{  // range格式可以对slice、map、数组、字符串等进行迭代循环
    …
}
```
for-each range循环示例：

```go
package main
import "fmt"

func main() {
        strings := []string{"google", "runoob"}
        for i, s := range strings {
                fmt.Println(i, s)
        }

        numbers := [6]int{1, 2, 3, 5} 
        for i,x:= range numbers {
                fmt.Printf("第 %d 位 x 的值 = %d\n", i,x)
        }  
}
```

输出结果为：

	0 google
	1 runoob
	第 0 位 x 的值 = 1
	第 1 位 x 的值 = 2
	第 2 位 x 的值 = 3
	第 3 位 x 的值 = 5
	第 4 位 x 的值 = 0
	第 5 位 x 的值 = 0

## 循环控制语句

![在这里插入图片描述](https://img-blog.csdnimg.cn/20200302173835951.png#pic_center)

break/continue语句可以使用标记跳出多层循环，示例如下：

```go
package main
import "fmt"

func main() {
    // 不使用标记
    fmt.Println("---- break ----")
    for i := 1; i <= 3; i++ {
        fmt.Printf("i: %d\n", i)
                for i2 := 11; i2 <= 13; i2++ {
                        fmt.Printf("i2: %d\n", i2)
                        break  // 跳出当前循环
                }
        }

    // 使用标记
    fmt.Println("---- break label ----")
    re:
        for i := 1; i <= 3; i++ {
            fmt.Printf("i: %d\n", i)
            for i2 := 11; i2 <= 13; i2++ {
                fmt.Printf("i2: %d\n", i2)
                break re  // 跳出标记的循环
            }
        }
}
```

 goto 语句可以无条件地转移到过程中指定的行。goto 语句通常与条件语句配合使用。可用来实现条件转移， 构成循环，跳出循环体等功能。但是，在结构化程序设计中一般不主张使用 goto 语句， 以免造成程序流程的混乱，使理解和调试程序都产生困难。

```go
goto label
label： …
// 示例
package main
import "fmt"

func main() {
   // 定义局部变量
   var a int = 10
   // 循环
   LOOP: for a < 20 {
      if a == 15 {
         a = a + 1
         goto LOOP
      }
      fmt.Printf("a的值为 : %d\n", a)
      a++     
   }  
}
```
# 函数

函数是基本的代码块，用于执行一个任务。Go 语言最少有个 main() 函数。你可以通过函数来划分不同功能，逻辑上每个函数执行的是指定的任务。函数声明告诉了编译器函数的名称，返回类型，和参数。函数可以返回多个值

Go 语言标准库提供了多种可动用的内置的函数。例如，len() 函数可以接受不同类型参数并返回该类型的长度。如果我们传入的是字符串则返回字符串的长度，如果传入的是数组，则返回数组中包含的元素个数。

Go 语言函数定义格式如下：

```go
func function_name( [parameter list] ) [return_types] {
   函数体
}
```
示例：

```go
// 按从大到小的顺序输出两个值
func sort(num1, num2 int) (int,int) {
   // 声明局部变量
   var result1,result2 int

   if (num1 > num2) {
      result1 = num1
      result2 = num2
   } else {
      result1 = num2
      result2 = num1
   }
   return result1, result2
}
```

### 函数参数
定义函数时参数列表中的参数a，在函数内部称之为形参，形参可以理解为在函数体内的局部变量。而我们调用这个函数时实际传给参数a的可能是变量b，变量b是实际参数。

调用函数，可以通过两种方式来传递参数，默认情况下使用值传递。

![在这里插入图片描述](https://img-blog.csdnimg.cn/20200302180533280.png#pic_center)

引用传递是指在调用函数时将实际参数的地址传递到函数中，那么在函数中对参数所进行的修改，将影响到实际参数。示例如下：

```go
/* 定义交换值函数*/
func swap(x *int, y *int) {
   var temp int
   temp = *x    /* 保持 x 地址上的值 */
   *x = *y      /* 将 y 值赋给 x */
   *y = temp    /* 将 temp 值赋给 y */
}
```

## 函数用法
- **函数作为实参**：Go 语言可以很灵活的创建函数，并作为另外一个函数的实参。示例如下：

```go
package main
import (
   "fmt"
   "math"
)

func main(){
   /* 声明函数变量 */
   getSquareRoot := func(x float64) float64 {
      return math.Sqrt(x)
   }
   
   /* 使用函数 */
   fmt.Println(getSquareRoot(9))
}
```

- **闭包**：Go 语言支持匿名函数，可作为闭包。匿名函数是一个"内联"语句或表达式。匿名函数的优越性在于可以直接使用函数内的变量，不必申明。示例如下：

```go
package main
import "fmt"

func getSequence() func() int {
   i:=0
   return func() int {
      i+=1
     return i  
   }
}

func main(){
   /* nextNumber 为一个函数，函数 i 为 0 */
   nextNumber := getSequence()  

   /* 调用 nextNumber 函数，i 变量自增 1 并返回 */
   fmt.Println(nextNumber())
   fmt.Println(nextNumber())
   fmt.Println(nextNumber())

   /* 创建新的函数 nextNumber1，并查看结果 */
   nextNumber1 := getSequence()  
   fmt.Println(nextNumber1())
   fmt.Println(nextNumber1())
}
```
- 方法：Go 语言中同时有函数和方法。一个方法就是一个包含了接受者的函数，接受者可以是命名类型或者结构体类型的一个值或者是一个指针。所有给定类型的方法属于该类型的方法集。语法格式如下：

```go
func (variable_name variable_data_type) function_name() [return_type]{
   /* 函数体*/
}
```
示例如下：

```go
package main
import "fmt"

/* 定义结构体 */
type Circle struct {
  radius float64
}

func main() {
  var c1 Circle
  c1.radius = 10.00
  fmt.Println("圆的面积 = ", c1.getArea())
}

// 注意这里是方法不是函数，该 method 属于 Circle 类型对象中的方法
func (c Circle) getArea() float64 {
  //c.radius 即为 Circle 类型对象中的属性
  return 3.14 * c.radius * c.radius
}
```
# 变量作用域

作用域为已声明标识符所表示的常量、类型、变量、函数或包在源代码中的作用范围。Go 语言中变量可以在三个地方声明：

- 函数内定义的变量称为局部变量：作用域只在函数体内，参数和返回值变量也是局部变量
- 函数外定义的变量称为全局变量：可以在整个包（任何函数中）甚至外部包（被导出后）使用；全局变量与局部变量名称可以相同，但是函数内的局部变量会被优先考虑
- 函数定义中的变量称为形式参数：作为函数的局部变量来使用

# 数组

## 一维数组

数组是具有相同唯一类型的一组已编号且长度固定的数据项序列，这种类型可以是任意的原始类型例如整形、字符串或者自定义类型。

相对于去声明 number0, number1, ..., number99 的变量，使用数组形式 numbers[0], numbers[1] ..., numbers[99] 更加方便且易于扩展。

数组元素可以通过索引（位置）来读取（或者修改），索引从 0 开始，第一个元素索引为 0，第二个索引为 1，以此类推。

```go
// 语法规则
var variable_name [SIZE] variable_type
// 示例
var balance [10] float32
var balance = [5]float32{1000.0, 2.0, 3.4, 7.0, 50.0}
// 如果忽略 [] 中的数字不设置数组大小，Go 语言会根据元素的个数来设置数组的大小
var balance = [...]float32{1000.0, 2.0, 3.4, 7.0, 50.0}
```

如果想向函数传递数组参数，则需要在函数定义时，声明形参为数组。

```go
// 形参设定数组大小
void myFunction(param [10]int){}
// 形参不设定数组大小
void myFunction(param []int){}
```

## 多维数组

```go
// 语法规则
var variable_name [SIZE1][SIZE2]...[SIZEN] variable_type
// 示例
a = [3][4]int{  
 {0, 1, 2, 3} ,   /*  第一行索引为 0 */
 {4, 5, 6, 7} ,   /*  第二行索引为 1 */
 {8, 9, 10, 11},   /* 第三行索引为 2 */ 
}
// 上例中倒数第二行必须有逗号，或将其写为如下形式
a = [3][4]int{  
 {0, 1, 2, 3} ,   /*  第一行索引为 0 */
 {4, 5, 6, 7} ,   /*  第二行索引为 1 */
 {8, 9, 10, 11}}   /* 第三行索引为 2 */
// 访问二维数组
val := a[2][3]
```

# 指针
一个指针变量指向了一个值的内存地址，在使用指针前你需要声明指针

```go
// 声明格式
var var_name *var-type
// 示例
var ip *int        /* 指向整型*/
var fp *float32    /* 指向浮点型 */
```
使用指针流程及示例：

- 定义指针变量
- 为指针变量赋值
- 访问指针变量中指向地址的值

```go
package main
import "fmt"

func main() {
   var a int= 20   /* 声明实际变量 */
   var ip *int        /* 声明指针变量 */

   ip = &a  /* 指针变量的存储地址 */

   fmt.Printf("a 变量的地址是: %x\n", &a  )

   /* 指针变量的存储地址 */
   fmt.Printf("ip 变量储存的指针地址: %x\n", ip )

   /* 使用指针访问值 */
   fmt.Printf("*ip 变量的值: %d\n", *ip )
}
```
输出结果为：

	a 变量的地址是: 20818a220
	ip 变量储存的指针地址: 20818a220
	*ip 变量的值: 20

## 空指针

当一个指针被定义后没有分配到任何变量时，它的值为 nil。nil 指针也称为空指针。nil在概念上和其它语言的null、None、nil、NULL一样，都指代零值或空值。一个指针变量通常缩写为 ptr。

```go
package main
import "fmt"

func main() {
   var  ptr *int

   fmt.Printf("ptr 的值为 : %x\n", ptr  )
}
```
输出结果为：

	ptr 的值为 : 0

```go
// 判断是否为空指针
if(ptr != nil)     /* ptr 不是空指针 */
if(ptr == nil)    /* ptr 是空指针 */
```

## 指针的更多用法

- 指针数组

```go
package main
import "fmt"

const MAX int = 3

func main() {
   a := []int{10,100,200}
   var i int
   var ptr [MAX]*int  // ptr 为整型指针数组，因此每个元素都指向了一个值

   for  i = 0; i < MAX; i++ {
      ptr[i] = &a[i] /* 整数地址赋值给指针数组 */
   }
   
   for  i = 0; i < MAX; i++ {
      fmt.Printf("a[%d] = %d\n", i,*ptr[i] )
   }
}
```

- 指向指针的指针：如果一个指针变量存放的又是另一个指针变量的地址，则称这个指针变量为指向指针的指针变量。当定义一个指向指针的指针变量时，第一个指针存放第二个指针的地址，第二个指针存放变量的地址
- 
![在这里插入图片描述](https://img-blog.csdnimg.cn/20200302191152595.png#pic_center)

```go
package main
import "fmt"

func main() {
   var a int
   var ptr *int
   var pptr **int

   a = 3000

   /* 指针 ptr 地址 */
   ptr = &a

   /* 指向指针 ptr 地址 */
   pptr = &ptr

   /* 获取 pptr 的值 */
   fmt.Printf("变量 a = %d\n", a )
   fmt.Printf("指针变量 *ptr = %d\n", *ptr )
   fmt.Printf("指向指针的指针变量 **pptr = %d\n", **pptr)
}
```

- 向函数传递指针

```go
package main
import "fmt"

func main() {
   /* 定义局部变量 */
   var a int = 100
   var b int= 200

   fmt.Printf("交换前 a 的值 : %d\n", a )
   fmt.Printf("交换前 b 的值 : %d\n", b )

   /* 调用函数用于交换值
   * &a 指向 a 变量的地址
   * &b 指向 b 变量的地址
   */
   swap(&a, &b);

   fmt.Printf("交换后 a 的值 : %d\n", a )
   fmt.Printf("交换后 b 的值 : %d\n", b )
}

func swap(x *int, y *int) {
   var temp int
   
   temp = *x    /* 保存 x 地址的值 */
   *x = *y      /* 将 y 赋值给 x */
   *y = temp    /* 将 temp 赋值给 y */
}
```

结果为：

交换前 a 的值 : 100
交换前 b 的值 : 200
交换后 a 的值 : 200
交换后 b 的值 : 100

# 结构体

Go 语言中数组可以存储同一类型的数据，但在结构体中我们可以为不同项定义不同的数据类型。结构体是由一系列具有相同类型或不同类型的数据构成的数据集合。

结构体定义需要使用 type 和 struct 语句。struct 语句定义一个新的数据类型，结构体中有一个或多个成员。type 语句设定了结构体的名称

```go
// 定义结构体
type struct_variable_type struct {
   member definition
   member definition
   ...
   member definition
}
// 将结构体用于变量的声明
variable_name := structure_variable_type {value1, value2...valuen}
variable_name := structure_variable_type { key1: value1, key2: value2..., keyn: valuen}  // 可以忽略部分字段
// 访问结构体成员
结构体.成员名
// 结构体指针
var struct_pointer *structure_variable_type
struct_pointer = &structure_variable_type1
struct_pointer.title
// 结构体可以用于函数参数，示例如下：
package main
import "fmt"

type Books struct {
   title string
   author string
   subject string
   book_id int
}

func main() {
   var Book1 Books        /* Declare Book1 of type Book */
   var Book2 Books        /* Declare Book2 of type Book */

   /* book 1 描述 */
   Book1.title = "Go 语言"
   Book1.author = "www.runoob.com"
   Book1.subject = "Go 语言教程"
   Book1.book_id = 6495407

   /* book 2 描述 */
   Book2.title = "Python 教程"
   Book2.author = "www.runoob.com"
   Book2.subject = "Python 语言教程"
   Book2.book_id = 6495700

   /* 打印 Book1 信息 */
   printBook(Book1)
   /* 打印 Book2 信息 */
   printBook2(&Book2)
}

func printBook( book Books ) {
   fmt.Printf( "Book title : %s\n", book.title)
   fmt.Printf( "Book author : %s\n", book.author)
   fmt.Printf( "Book subject : %s\n", book.subject)
   fmt.Printf( "Book book_id : %d\n", book.book_id)
}

func printBook2( book *Books ) {
   fmt.Printf( "Book title : %s\n", book.title)
   fmt.Printf( "Book author : %s\n", book.author)
   fmt.Printf( "Book subject : %s\n", book.subject)
   fmt.Printf( "Book book_id : %d\n", book.book_id)
}
```

# 切片Slice

切片是对数组的抽象。Go 数组的长度不可改变，在特定场景中这样的集合就不太适用，Go中提供了一种灵活，功能强悍的内置类型切片("动态数组"),与数组相比切片的长度是不固定的，可以追加元素，在追加时可能使切片的容量增大。

```go
// 定义切片
// 通过声明一个未指定大小的数组来定义切片
var identifier []type
// 使用make()函数来创建贴片
var slice1 []type = make([]type, len)
slice1 := make([]type, len)
slice1 := make([]type, length, capacity)  // 也可以指定容量，其中capacity为可选参数。

// 切片初始化
// 直接初始化
s :=[] int {1,2,3}
// 通过数组初始化
s := arr[:]
s := arr[startIndex:endIndex]
s := arr[startIndex:]
s := arr[:endIndex]
// 通过其他切片初始化
s1 := s[startIndex:endIndex]
// 通过内置函数make()初始化
s := make([]int,len,cap)

// len()和cap()函数
// len()获取长度，cap()获取最大长度，示例如下
package main
import "fmt"

func main() {
   var numbers = make([]int,3,5)

   printSlice(numbers)
}

func printSlice(x []int){
   fmt.Printf("len=%d cap=%d slice=%v\n",len(x),cap(x),x)
}
```

输出结果为：

	len=3 cap=5 slice=[0 0 0]

```go
// 空（nil）切片
// 一个切片在未初始化之前默认为 nil，长度为 0，示例如下
package main
import "fmt"

func main() {
   var numbers []int

   printSlice(numbers)

   if(numbers == nil){
      fmt.Printf("切片是空的")
   }
}

func printSlice(x []int){
   fmt.Printf("len=%d cap=%d slice=%v\n",len(x),cap(x),x)
}
```
输出结果为：

	len=0 cap=0 slice=[]
	切片是空的

```go
// append()和copy()函数
// 如果想增加切片的容量，我们必须创建一个新的更大的切片并把原分片的内容都拷贝过来
package main
import "fmt"

func main() {
   var numbers []int
   printSlice(numbers)

   /* 允许追加空切片 */
   numbers = append(numbers, 0)
   printSlice(numbers)

   /* 向切片添加一个元素 */
   numbers = append(numbers, 1)
   printSlice(numbers)

   /* 同时添加多个元素 */
   numbers = append(numbers, 2,3,4)
   printSlice(numbers)

   /* 创建切片 numbers1 是之前切片的两倍容量*/
   numbers1 := make([]int, len(numbers), (cap(numbers))*2)

   /* 拷贝 numbers 的内容到 numbers1 */
   copy(numbers1,numbers)
   printSlice(numbers1)   
}

func printSlice(x []int){
   fmt.Printf("len=%d cap=%d slice=%v\n",len(x),cap(x),x)
}
```

输出结果为：

	len=0 cap=0 slice=[]
	len=1 cap=1 slice=[0]
	len=2 cap=2 slice=[0 1]
	len=5 cap=6 slice=[0 1 2 3 4] ==为什么这里cap=6？==
	len=5 cap=12 slice=[0 1 2 3 4]


​	
# 范围Range
range 关键字用于 for 循环中迭代数组(array)、切片(slice)、通道(channel)或集合(map)的元素。在数组和切片中它返回元素的索引和索引对应的值，在集合中返回 key-value 对。示例如下：

```go
package main
import "fmt"

func main() {
    //这是我们使用range去求一个slice的和。使用数组跟这个很类似
    nums := []int{2, 3, 4}
    sum := 0
    
    for _, num := range nums {
        sum += num
    }
    fmt.Println("sum:", sum)

    //在数组上使用range将传入index和值两个变量。上面那个例子我们不需要使用该元素的序号，所以我们使用空白符"_"省略了。有时侯我们确实需要知道它的索引。
    for i, num := range nums {
        if num == 3 {
            fmt.Println("index:", i)
        }
    }

    //range也可以用在map的键值对上。
    kvs := map[string]string{"a": "apple", "b": "banana"}
    for k, v := range kvs {
        fmt.Printf("%s -> %s\n", k, v)
    }
    
    //range也可以用来枚举Unicode字符串。第一个参数是字符的索引，第二个是字符（Unicode的值）本身。
    for i, c := range "go" {
        fmt.Println(i, c)
    }
}
```
输出结果为：

	sum: 9
	index: 1
	a -> apple
	b -> banana
	0 103
	1 111

# 集合Map
Map 是一种无序的键值对的集合。Map 最重要的一点是通过 key 来快速检索数据，key 类似于索引，指向数据的值。

Map 是一种集合，所以我们可以像迭代数组和切片那样迭代它。不过，Map 是无序的，我们无法决定它的返回顺序，这是因为 Map 是使用 hash 表来实现的。

```go
// 定义map
// 声明变量，默认 map 是 nil
var map_variable map[key_data_type]value_data_type
// 使用make函数创建一个非nil的map，nil map不能赋值
map_variable := make(map[key_data_type]value_data_type)
// 示例如下
package main
import "fmt"

func main() {
    var countryCapitalMap map[string]string /*创建集合 */
    countryCapitalMap = make(map[string]string)

    /* map插入key - value对,各个国家对应的首都 */
    countryCapitalMap [ "France" ] = "巴黎"
    countryCapitalMap [ "Italy" ] = "罗马"
    countryCapitalMap [ "Japan" ] = "东京"
    countryCapitalMap [ "India " ] = "新德里"

    /*使用键输出地图值 */ 
    for country := range countryCapitalMap {
        fmt.Println(country, "首都是", countryCapitalMap [country])
    }

    /*查看元素在集合中是否存在，【注意】此处左侧两个变量，第一个是值，第二个是是否存在 */
    capital, ok := countryCapitalMap [ "American" ] /*如果确定是真实的,则存在,否则不存在 */
    /*fmt.Println(capital) */
    /*fmt.Println(ok) */
    if (ok) {
        fmt.Println("American 的首都是", capital)
    } else {
        fmt.Println("American 的首都不存在")
    }
}
```
输出结果为：

	France 首都是 巴黎
	Italy 首都是 罗马
	Japan 首都是 东京
	India  首都是 新德里
	American 的首都不存在

```go
// delete()函数
// 用于删除集合的元素, 参数为 map 和其对应的 key
delete(countryCapitalMap, "France")
```
# 递归函数
递归，就是在运行的过程中调用自己。Go 语言支持递归。但我们在使用递归时，开发者需要设置退出条件，否则递归将陷入无限循环中。递归函数对于解决数学上的问题是非常有用的，就像计算阶乘，生成斐波那契数列等。

```c
// 语法格式
func recursion() {
   recursion() /* 函数调用自身 */
}

func main() {
   recursion()
}
// 示例-阶乘：
package main
import "fmt"

func Factorial(n uint64)(result uint64) {
    if (n > 0) {
        result = n * Factorial(n-1)
        return result
    }
    return 1
}

func main() {  
    var i int = 15
    fmt.Printf("%d 的阶乘是 %d\n", i, Factorial(uint64(i)))
}
// 示例-斐波那契数列
package main
import "fmt"

func fibonacci(n int) int {
  if n < 2 {
   return n
  }
  return fibonacci(n-2) + fibonacci(n-1)
}

func main() {
    var i int
    for i = 0; i < 10; i++ {
       fmt.Printf("%d\t", fibonacci(i))
    }
}
```

# 类型转换

类型转换用于将一种数据类型的变量转换为另外一种类型的变量。

```go
// 语法格式
// type_name 为类型，expression 为表达式
type_name(expression)
// 示例
package main
import "fmt"

func main() {
   var sum int = 17
   var count int = 5
   var mean float32

   mean = float32(sum)/float32(count)
   fmt.Printf("mean 的值为: %f\n",mean)
}
```
输出结果为：
	mean 的值为: 3.400000

# 接口
Go 语言提供了另外一种数据类型即接口，它把所有的具有共性的方法定义在一起，任何其他类型只要实现了这些方法就是实现了这个接口。

```go
// 语法格式
/* 定义接口 */
type interface_name interface {
   method_name1 [return_type]
   method_name2 [return_type]
   method_name3 [return_type]
   ...
   method_namen [return_type]
}

/* 定义结构体 */
type struct_name struct {
   /* variables */
}

/* 实现接口方法 */
func (struct_name_variable struct_name) method_name1() [return_type] {
   /* 方法实现 */
}
...
func (struct_name_variable struct_name) method_namen() [return_type] {
   /* 方法实现*/
}
// 示例
package main
import "fmt"

type Phone interface {
    call()
}

type NokiaPhone struct {
}

func (nokiaPhone NokiaPhone) call() {
    fmt.Println("I am Nokia, I can call you!")
}

type IPhone struct {
}

func (iPhone IPhone) call() {
    fmt.Println("I am iPhone, I can call you!")
}

func main() {
    var phone Phone

    phone = new(NokiaPhone)
    phone.call()

    phone = new(IPhone)
    phone.call()
}
```
我们定义了一个接口Phone，接口里面有一个方法call()。然后我们在main函数里面定义了一个Phone类型变量，并分别为之赋值为NokiaPhone和IPhone。然后调用call()方法

输出结果为：

	I am Nokia, I can call you!
	I am iPhone, I can call you!

# 错误处理
Go 语言通过内置的错误接口提供了非常简单的错误处理机制，error类型是一个接口类型，定义如下：

```go
type error interface {
    Error() string
}
```
我们可以在编码中通过实现 error 接口类型来生成错误信息，函数通常在最后的返回值中返回错误信息。使用errors.New 可返回一个错误信息。在下面的例子中，我们在调用Sqrt的时候传递的一个负数，然后就得到了non-nil的error对象，将此对象与nil比较，结果为true，所以fmt.Println(fmt包在处理error时会调用Error方法)被调用，以输出错误

```go
func Sqrt(f float64) (float64, error) {
    if f < 0 {
        return 0, errors.New("math: square root of negative number")
    }
}

result, err:= Sqrt(-1)

if err != nil {
   fmt.Println(err)
}
```
示例如下：

```go
package main
import "fmt"

// 定义一个 DivideError 结构
type DivideError struct {
    dividee int
    divider int
}

// 实现 `error` 接口
func (de *DivideError) Error() string {
    strFormat := `
    Cannot proceed, the divider is zero.
    dividee: %d
    divider: 0
`
    return fmt.Sprintf(strFormat, de.dividee)
}

// 定义 `int` 类型除法运算的函数
func Divide(varDividee int, varDivider int) (result int, errorMsg string) {
    if varDivider == 0 {
            dData := DivideError{
                    dividee: varDividee,
                    divider: varDivider,
            }
            errorMsg = dData.Error()
            return
    } else {
            return varDividee / varDivider, ""
    }
}

func main() {

    // 正常情况
    if result, errorMsg := Divide(100, 10); errorMsg == "" {
            fmt.Println("100/10 = ", result)
    }

    // 当除数为零的时候会返回错误信息
    if _, errorMsg := Divide(100, 0); errorMsg != "" {
            fmt.Println("errorMsg is: ", errorMsg)
    }
}
```
输出结果为：
	100/10 =  10
	errorMsg is:  
	    Cannot proceed, the divider is zero.
	    dividee: 100
	    divider: 0
	
# 并发

Go 语言支持并发，我们只需要通过 go 关键字来开启 goroutine 即可。goroutine 是轻量级线程，goroutine 的调度是由 Golang 运行时进行管理的。

Go 允许使用 go 语句开启一个新的运行期线程， 即 goroutine，以一个不同的、新创建的 goroutine 来执行一个函数。 同一个程序中的所有 goroutine 共享同一个地址空间。

```go
// 语法格式
go 函数名(参数列表)
// 示例
package main

import (
        "fmt"
        "time"
)

func say(s string) {
        for i := 0; i < 5; i++ {
                time.Sleep(100 * time.Millisecond)
                fmt.Println(s)
        }
}

func main() {
        go say("world")
        say("hello")
}
```
输出结果为：
	world
	hello
	hello
	world
	world
	hello
	hello
	world
	world
	hello

输出的 hello 和 world 是没有固定先后顺序，因为它们是两个 goroutine 在执行

## 通道channel
通道（channel）是用来传递数据的一个数据结构。通道可用于两个 goroutine 之间通过传递一个指定类型的值来同步运行和通讯。操作符 <- 用于指定通道的方向，发送或接收。如果未指定方向，则为双向通道。

```go
// 声明通道，通道在使用前必须声明
ch := make(chan int)
// 使用通道
ch <- v    // 把 v 发送到通道 ch
v := <-ch  // 从 ch 接收数据
           // 并把值赋给 v
```
- **不带缓冲区的通道**：默认情况下，通道是不带缓冲区的。发送端发送数据，同时必须有接收端相应的接收数据

```go
package main
import "fmt"

func sum(s []int, c chan int) {
        sum := 0
        for _, v := range s {
                sum += v
        }
        c <- sum // 把 sum 发送到通道 c
}

func main() {
        s := []int{7, 2, 8, -9, 4, 0}

        c := make(chan int)
        go sum(s[:len(s)/2], c)
        go sum(s[len(s)/2:], c)
        x, y := <-c, <-c // 从通道 c 中接收

        fmt.Println(x, y, x+y)
}
```

- **带缓冲区的通道**：通道可以设置缓冲区，通过 make 的第二个参数指定缓冲区大小。

```go
ch := make(chan int,100)
```

带缓冲区的通道允许发送端的数据发送和接收端的数据获取处于异步状态，就是说发送端发送的数据可以放在缓冲区里面，可以等待接收端去获取数据，而不是立刻需要接收端去获取数据。不过由于缓冲区的大小是有限的，所以还是必须有接收端来接收数据的，否则缓冲区一满，数据发送端就无法再发送数据了。

如果通道不带缓冲，发送方会阻塞直到接收方从通道中接收了值。如果通道带缓冲，发送方则会阻塞直到发送的值被拷贝到缓冲区内；如果缓冲区已满，则意味着需要等待直到某个接收方获取到一个值。接收方在有值可以接收之前会一直阻塞。

```go
package main
import "fmt"

func main() {
    // 这里我们定义了一个可以存储整数类型的带缓冲通道
        // 缓冲区大小为2
        ch := make(chan int, 2)
        
        // 因为 ch 是带缓冲的通道，我们可以同时发送两个数据
        // 而不用立刻需要去同步读取数据
        ch <- 1
        ch <- 2

        // 获取这两个数据
        fmt.Println(<-ch)
        fmt.Println(<-ch)
}
```
输出结果为：

	1
	2

- 遍历通道与挂机通道：Go 通过 range 关键字来实现遍历读取到的数据，类似于与数组或切片，如果通道接收不到数据后 ok 就为 false，这时通道就可以使用 close() 函数来关闭。

```go
// 语法规则
v, ok := <- ch
// 示例
package main
import "fmt"

func fibonacci(n int, c chan int) {
        x, y := 0, 1
        
        for i := 0; i < n; i++ {
                c <- x
                x, y = y, x+y
        }
        close(c)
}

func main() {
        c := make(chan int, 10)
        
        go fibonacci(cap(c), c)
        
        // range 函数遍历每个从通道接收到的数据，因为 c 在发送完 10 个
        // 数据之后就关闭了通道，所以这里我们 range 函数在接收到 10 个数据
        // 之后就结束了。如果上面的 c 通道不关闭，那么 range 函数就不
        // 会结束，从而在接收第 11 个数据的时候就阻塞了。
        for i := range c {
                fmt.Println(i)
        }
}
```
输出结果为：
	0
	1
	1
	2
	3
	5
	8
	13
	21
	34
	
# 开发工具

- GoLand（30天免费试用期）
- LiteIDE（开源）
- Eclipse

[具体介绍](https://www.runoob.com/go/go-ide.html)

# 补充链接
- [基本的http请求](https://www.cnblogs.com/Paul-watermelon/p/11386392.html)
- [格式化输出 Printf](https://www.cnblogs.com/Paul-watermelon/p/10934020.html)
