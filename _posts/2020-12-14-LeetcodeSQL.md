---
title: 力扣刷题 | SQL语句例题及代码
author: 钟欣然
date: 2020-12-12 00:46:00 +0800
categories: [力扣刷题, 按问题分类]
math: true
mermaid: true
---

本文内容来自 [leetcode](https://leetcode-cn.com/problemset/database/)

SQL 基础教程参考 [w3school](https://www.w3school.com.cn/sql/index.asp)

# 175 组合两个表 
> 编写一个 SQL 查询，满足条件：无论 person 是否有地址信息，都需要基于上述两表提供 person 的以下信息：
FirstName, LastName, City, State

![在这里插入图片描述](https://img-blog.csdnimg.cn/20200521201749941.png)

**解法：** 左连接

```sql
SELECT FirstName,
       LastName,
       City,
       STATE
FROM Person
LEFT JOIN Address ON Person.PersonId = Address.PersonId
```

# 176 第二高的薪水
> 编写一个 SQL 查询，获取 Employee 表中第二高的薪水（Salary）。

![在这里插入图片描述](https://img-blog.csdnimg.cn/20200521202708611.png)

> 例如上述 Employee 表，SQL查询应该返回 200 作为第二高的薪水。如果不存在第二高的薪水，那么查询应返回 null。

![在这里插入图片描述](https://img-blog.csdnimg.cn/20200521202742535.png)

**排名的三种方法：**

- 300 200 200 100 为 1 2 3 4
- 300 200 200 100 为 1 2 2 4
- 300 200 200 100 为 1 2 2 3

本题为第三种

**解法1：** 嵌套查询，先查出最大值，再查出小于最大值里的最大值

```sql
SELECT max(Salary) AS SecondHighestSalary
FROM Employee
WHERE Salary <
    (SELECT max(Salary)
     FROM Employee)
```

**解法2：** 采用 limit 和 offset 方法，外层是为了保证只有一条数据时可以返回空值

```sql
SELECT
    (SELECT DISTINCT
            Salary
        FROM
            Employee
        ORDER BY Salary DESC
        LIMIT 1 OFFSET 1) AS SecondHighestSalary
```

# 177 第N高的薪水
> 编写一个 SQL 查询，获取 Employee 表中第 n 高的薪水（Salary）。

![在这里插入图片描述](https://img-blog.csdnimg.cn/20200521204940718.png)

> 例如上述 Employee 表，n = 2 时，应返回第二高的薪水 200。如果不存在第 n 高的薪水，那么查询应返回 null。

![在这里插入图片描述](https://img-blog.csdnimg.cn/20200521205003728.png)

**解法1：** limit 和 offset 方法

```sql
CREATE FUNCTION getNthHighestSalary(N INT) RETURNS INT
BEGIN
    Set N := N-1;
  RETURN (
      SELECT Salary
      FROM Employee
      GROUP BY Salary
      ORDER BY Salary DESC LIMIT 1
      OFFSET N
  );
END
```

**解法2：** 排名第N的薪水意味着该表中存在N-1个比其更高的薪水

```sql
CREATE FUNCTION getNthHighestSalary(N INT) RETURNS INT
BEGIN
  RETURN (
      SELECT DISTINCT salary
      FROM Employee e
      WHERE
          (SELECT count(DISTINCT salary)
          FROM Employee
          WHERE salary > e.salary) = N-1
  );
END
```

**解法3：** 自连接，把每个薪水和大于等于这个薪水的值连起来，按照第一个薪水分组，找到具有 N 个不同的第二个薪水的值的第一个薪水

```sql
CREATE FUNCTION getNthHighestSalary(N INT) RETURNS INT
BEGIN
  RETURN (
      SELECT DISTINCT e1.salary
      FROM Employee e1
      LEFT JOIN Employee e2 ON e1.salary <= e2.salary
      GROUP BY e1.salary HAVING count(DISTINCT e2.salary) = N
  );
END
```

**解法4：** 笛卡尔积，和解法3相比是用子查询替代了连接

```sql
CREATE FUNCTION getNthHighestSalary(N INT) RETURNS INT
BEGIN
  RETURN (
      SELECT DISTINCT e1.salary
      FROM Employee e1, Employee e2
      where e1.salary <= e2.salary
      GROUP BY e1.salary HAVING count(DISTINCT e2.salary) = N
  );
END
```

**解法5：** 定义变量

```sql
CREATE FUNCTION getNthHighestSalary(N INT) RETURNS INT
BEGIN
  RETURN (
      SELECT DISTINCT salary
      FROM
      (SELECT salary, @r:=IF(@p=salary, @r, @r+1) AS rnk, @p:=salary
      FROM Employee,
          (SELECT @r:=0, @p:= NULL) init
      ORDER BY salary DESC) tmp
      WHERE rnk = N
  );
END
```

**解法6：** 窗口函数，在 mysql8.0 中有相关的内置函数，而且考虑了各种排名问题：

- row_number(): 同薪不同名，相当于行号，例如3000、2000、2000、1000排名后为1、2、3、4
- rank(): 同薪同名，有跳级，例如3000、2000、2000、1000排名后为1、2、2、4
- dense_rank(): 同薪同名，无跳级，例如3000、2000、2000、1000排名后为1、2、2、3
- ntile(): 分桶排名，即首先按桶的个数分出第一二三桶，然后各桶内从1排名，实际不是很常用

前三个函数必须要要与其搭档over()配套使用，over()中的参数常见的有两个，分别是

- partition by，按某字段切分
- order by，与常规order by用法一致，也区分ASC(默认)和DESC，因为排名总得有个依据

```sql
CREATE FUNCTION getNthHighestSalary(N INT) RETURNS INT
BEGIN
  RETURN (
	  SELECT DISTINCT salary
	  FROM
	    (SELECT salary,
	            dense_rank() over(
	                              ORDER BY salary DESC) AS rnk
	     FROM employee) tmp
	  WHERE rnk = N
  );
END
```

综上，总结MySQL查询的一般性思路是：

- 能用单表优先用单表，即便是需要用 group by、order by、limit 等，效率一般也比多表高
- 不能用单表时优先用连接，连接是SQL中非常强大的用法，小表驱动大表+建立合适索引+合理运用连接条件，基本上连接可以解决绝大部分问题。但join级数不宜过多，毕竟是一个接近指数级增长的关联效果
- 能不用子查询、笛卡尔积尽量不用，虽然很多情况下MySQL优化器会将其优化成连接方式的执行过程，但效率仍然难以保证
- 自定义变量在复杂SQL实现中会很有用，例如LeetCode中困难级别的数据库题目很多都需要借助自定义变量实现
- 如果MySQL版本允许，某些带聚合功能的查询需求应用窗口函数是一个最优选择。

![在这里插入图片描述](https://img-blog.csdnimg.cn/20200521231059348.png)

# 178 分数排名

> 编写一个 SQL 查询来实现分数排名。
如果两个分数相同，则两个分数排名（Rank）相同。请注意，平分后的下一个名次应该是下一个连续的整数值。换句话说，名次之间不应该有“间隔”。

![在这里插入图片描述](https://img-blog.csdnimg.cn/20200521233554680.png)

> 例如，根据上述给定的 Scores 表，你的查询应该返回（按分数从高到低排列）：

![在这里插入图片描述](https://img-blog.csdnimg.cn/20200521233618255.png)

**解法1：** 定义变量，注意 Rank 必须使用 \`Rank\`，否则会报错


```sql
SELECT Score,
       rnk+0 AS `Rank`
FROM
  (SELECT Score, @r:=IF(@s=Score,@r,@r+1) AS rnk, @s:=Score
   FROM Scores,
     (SELECT @r:=0,@s:=NULL) init
   ORDER BY Score DESC) tmp
```

**解法2：** 常规解法，嵌套查询

```sql
SELECT a.Score AS Score,

  (SELECT count(DISTINCT b.Score)
   FROM Scores b
   WHERE b.Score >= a.Score) AS `Rank`
FROM Scores a
ORDER BY a.Score DESC
```

# 180 连续出现的数字
> 编写一个 SQL 查询，查找所有至少连续出现三次的数字。

![在这里插入图片描述](https://img-blog.csdnimg.cn/20200522201258774.png)

> 例如，给定上面的 Logs 表， 1 是唯一连续出现至少三次的数字。

![在这里插入图片描述](https://img-blog.csdnimg.cn/20200522201323876.png)

**解法：** 三次连接，取相同的数字

```sql
SELECT DISTINCT l1.num AS ConsecutiveNums
FROM logs l1
LEFT JOIN logs l2 ON l1.Id = l2.Id-1
LEFT JOIN logs l3 ON l1.Id = l3.Id-2
WHERE l1.num = l2.num
  AND l1.num = l3.num
```

# 181 超过经理收入的员工
> Employee 表包含所有员工，他们的经理也属于员工。每个员工都有一个 Id，此外还有一列对应员工的经理的 Id。
给定 Employee 表，编写一个 SQL 查询，该查询可以获取收入超过他们经理的员工的姓名。在上面的表格中，Joe 是唯一一个收入超过他的经理的员工。

![在这里插入图片描述](https://img-blog.csdnimg.cn/2020052220184885.png)

![在这里插入图片描述](https://img-blog.csdnimg.cn/20200522201908369.png)

**解法：** 连接

```sql
SELECT e1.name AS Employee
FROM employee e1
LEFT JOIN employee e2 ON e1.managerId = e2.Id
WHERE e1.salary > e2.salary
```

# 182 查找重复的电子邮箱
> 编写一个 SQL 查询，查找 Person 表中所有重复的电子邮箱。

![在这里插入图片描述](https://img-blog.csdnimg.cn/20200522203048623.png)

> 根据以上输入，你的查询应返回以下结果：

![在这里插入图片描述](https://img-blog.csdnimg.cn/20200522203107866.png)

**解法：** group by 和 having
```sql
SELECT email
FROM person
GROUP BY email HAVING count(*) > 1
```

# 183 从不订购的客户
> 某网站包含两个表，Customers 表和 Orders 表。编写一个 SQL 查询，找出所有从不订购任何东西的客户。

![在这里插入图片描述](https://img-blog.csdnimg.cn/20200522203603421.png)

>例如给定上述表格，你的查询应返回：

![在这里插入图片描述](https://img-blog.csdnimg.cn/20200522203621954.png)

**解法1：** 连接

```sql
SELECT name AS customers
FROM customers
LEFT JOIN orders ON customers.id = orders.customerId
WHERE orders.id IS NULL
```

**解法2：** where 子句查询

```sql
SELECT name AS customers
FROM customers
WHERE id NOT IN
    (SELECT customerId
     FROM orders)
```

# 184 部门工资最高的员工

> Employee 表包含所有员工信息，每个员工有其对应的 Id, salary 和 department Id。
Department 表包含公司所有部门的信息。
编写一个 SQL 查询，找出每个部门工资最高的员工。例如，根据上述给定的表格，Max 在 IT 部门有最高工资，Henry 在 Sales 部门有最高工资。

![在这里插入图片描述](https://img-blog.csdnimg.cn/20200522205855812.png)

![在这里插入图片描述](https://img-blog.csdnimg.cn/20200522205907671.png)

![在这里插入图片描述](https://img-blog.csdnimg.cn/20200522205921170.png)

**解法：** 先找到每个部门的最高工资，再连接表查出最高工资员工的名字等信息，注意最高工资可能不止一个人

**注意：** join = inner join

```sql
SELECT tmp.Department,
       tmp.Employee,
       tmp.Salary
FROM
  (SELECT d.name AS Department,
          e.name AS Employee,
          e.salary AS Salary,
          e.departmentId AS DepartmentId
   FROM employee AS e
   JOIN department d ON e.departmentId = d.id) tmp
WHERE tmp.Salary =
    (SELECT max(salary)
     FROM employee e2
     WHERE tmp.DepartmentId = e2.DepartmentId)
```

# 185 部门工资前三高的员工
> Employee 表包含所有员工信息，每个员工有其对应的工号 Id，姓名 Name，工资 Salary 和部门编号 DepartmentId 。
Department 表包含公司所有部门的信息。

![在这里插入图片描述](https://img-blog.csdnimg.cn/20200522220726853.png)

![在这里插入图片描述](https://img-blog.csdnimg.cn/2020052222074235.png)

> 编写一个 SQL 查询，找出每个部门获得前三高工资的所有员工。例如，根据上述给定的表，查询结果应返回：
> 
![在这里插入图片描述](https://img-blog.csdnimg.cn/20200522220755461.png)

**解法1：** 仿照上题解法，先找到每个部门的前三高工资，再连接表查出最高工资员工的名字等信息

```sql
SELECT tmp.Department,
       tmp.Employee,
       tmp.Salary
FROM
  (SELECT d.name AS Department,
          e.name AS Employee,
          e.salary AS Salary,
          e.departmentId AS DepartmentId
   FROM employee AS e
   JOIN department d ON e.departmentId = d.id) tmp
WHERE tmp.Salary IN
    (SELECT DISTINCT salary
     FROM employee e2
     WHERE tmp.DepartmentId = e2.DepartmentId
     ORDER BY salary DESC LIMIT 3)
```

**解法2：** 和解法1相似，最后判断条件为所在部门高于这一薪水的不同薪水数量小于3

```sql
SELECT d.name AS Department,
       e.name AS Employee,
       e.salary AS Salary
FROM employee AS e
JOIN department d ON e.departmentId = d.id
WHERE
    (SELECT count(DISTINCT e1.salary)
     FROM employee e1
     WHERE e1.departmentId = e.departmentId
       AND e1.salary > e.salary) < 3
```

**解法3：** 采用窗口函数

```sql
SELECT tmp.Department,
       tmp.Employee,
       tmp.Salary
FROM
  (SELECT d.name AS Department,
          e.name AS Employee,
          e.salary AS Salary,
          e.departmentId AS DepartmentId,
          dense_rank() OVER(PARTITION BY e.departmentId
                  ORDER BY e.salary DESC) AS Rnk
   FROM employee AS e
   JOIN department d ON e.departmentId = d.id) tmp
WHERE tmp.Rnk <= 3
```

# 196 删除重复的电子邮箱

> 编写一个 SQL 查询，来删除 Person 表中所有重复的电子邮箱，重复的邮箱里只保留 Id 最小 的那个。

![在这里插入图片描述](https://img-blog.csdnimg.cn/20200522223427195.png)

> 例如，在运行你的查询语句之后，上面的 Person 表应返回以下几行：

![在这里插入图片描述](https://img-blog.csdnimg.cn/20200522223440113.png)

> 提示：
> - 执行 SQL 之后，输出是整个 Person 表
> - 使用 delete 语句

**解法：** 删除多表中的一个表

```sql
DELETE p1
FROM person p1,
     person p2
WHERE p1.id > p2.id
  AND p1.email = p2.email
```

# 197 上升的温度
> 给定一个 Weather 表，编写一个 SQL 查询，来查找与之前（昨天的）日期相比温度更高的所有日期的 Id。

![在这里插入图片描述](https://img-blog.csdnimg.cn/20200522224502595.png)

> 例如，根据上述给定的 Weather 表格，返回如下 Id：

![在这里插入图片描述](https://img-blog.csdnimg.cn/20200522224524945.png)

**解法：** 自连接，时间差函数用 datediff，前面的减后面的

```sql
SELECT w1.id
FROM weather w1
JOIN weather w2 ON DATEDIFF(w1.recordDate, w2.recordDate) = 1
WHERE w1.temperature > w2.temperature
```

# 262 行程和用户
>  Trips 表中存所有出租车的行程信息。每段行程有唯一键 Id，Client_Id 和 Driver_Id 是 Users 表中 Users_Id 的外键。Status 是枚举类型，枚举成员为 (‘completed’, ‘cancelled_by_driver’, ‘cancelled_by_client’)。
Users 表存所有用户。每个用户有唯一键 Users_Id。Banned 表示这个用户是否被禁止，Role 则是一个表示（‘client’, ‘driver’, ‘partner’）的枚举类型。

![在这里插入图片描述](https://img-blog.csdnimg.cn/20200522231915624.png)

![在这里插入图片描述](https://img-blog.csdnimg.cn/20200522231951365.png)

> 写一段 SQL 语句查出 2013年10月1日 至 2013年10月3日 期间非禁止用户的取消率。基于上表，你的 SQL 语句应返回如下结果，取消率（Cancellation Rate）保留两位小数。
取消率的计算方式如下：(被司机或乘客取消的非禁止用户生成的订单数量) / (非禁止用户生成的订单总数)

![在这里插入图片描述](https://img-blog.csdnimg.cn/20200522232018418.png)

**解法：** 连接后判断，再分组展示

**注意：**

- 直接取逻辑表达式返回 0/1 
- 时间前后判断可以使用 between and 或者 >、<
- group by 后面不能再有 where，只能有 having 对分组后结果进行判断

```sql
SELECT t.request_at AS `Day`,
       round(1-avg(status = 'completed'), 2) AS `Cancellation Rate`
FROM trips t
LEFT JOIN users u1 ON t.client_id = u1.users_id
LEFT JOIN users u2 ON t.driver_id = u2.users_id
WHERE u1.banned = 'No'
  AND u2.banned = 'No'
  AND t.request_at BETWEEN '2013-10-01' AND '2013-10-03'
GROUP BY t.request_at
```

# 595 大的国家
> 这里有张 World 表
如果一个国家的面积超过300万平方公里，或者人口超过2500万，那么这个国家就是大国家。
编写一个SQL查询，输出表中所有大国家的名称、人口和面积。

![在这里插入图片描述](https://img-blog.csdnimg.cn/20200522232942983.png)

> 例如，根据上表，我们应该输出:

![在这里插入图片描述](https://img-blog.csdnimg.cn/20200522232958951.png)


**解法：** where 查询

```python
SELECT name,
       population,
       area
FROM world
WHERE area > 3000000
  OR population > 25000000
```

# 596 超过5名学生的课
> 有一个courses 表 ，有: student (学生) 和 class (课程)。
> 请列出所有超过或等于5名学生的课。

![在这里插入图片描述](https://img-blog.csdnimg.cn/20200522233640775.png)

![在这里插入图片描述](https://img-blog.csdnimg.cn/20200522233653141.png)

> 注意：
>
> * 学生在每个课中不应被重复计算

**解法：** group by 和 having
```sql
SELECT class
FROM courses
GROUP BY class HAVING count(DISTINCT student) >= 5
```
#### 601 体院馆的人流量
> X 市建了一个新的体育馆，每日人流量信息被记录在这三列信息中：序号 (id)、日期 (visit_date)、 人流量 (people)。
> 请编写一个查询语句，找出人流量的高峰期。高峰期时，至少连续三行记录中的人流量不少于100。
> 例如，表 stadium：

![在这里插入图片描述](https://img-blog.csdnimg.cn/20200522235132379.png)

> 对于上面的示例数据，输出为：

![在这里插入图片描述](https://img-blog.csdnimg.cn/20200522235147806.png)

> 提示：
>
> - 每天只有一行记录，日期随着 id 的增加而增加。

**解法1**：对每一行进行判断，每一行均可能为高峰期的第一天、第二天、第三天

```sql
SELECT *
FROM stadium s
WHERE s.people >= 100
  AND (((
           (SELECT people
            FROM stadium s1
            WHERE s1.id = s.id - 2) >= 100)
        AND (
               (SELECT people
                FROM stadium s2
                WHERE s2.id = s.id - 1) >= 100))
       OR ((
              (SELECT people
               FROM stadium s1
               WHERE s1.id = s.id - 1) >= 100)
           AND (
                  (SELECT people
                   FROM stadium s2
                   WHERE s2.id = s.id + 1) >= 100))
       OR ((
              (SELECT people
               FROM stadium s1
               WHERE s1.id = s.id + 1) >= 100)
           AND (
                  (SELECT people
                   FROM stadium s2
                   WHERE s2.id = s.id + 2) >= 100)))
```

**解法2：** 自连接，分别检查是否为第一天、第二天、第三天

```sql
SELECT DISTINCT s1.*
FROM stadium s1,
     stadium s2,
     stadium s3
WHERE s1.people >= 100
  AND s2.people >= 100
  AND s3.people >= 100
  AND ((s1.id = s2.id - 1
        AND s1.id = s3.id -2)
       OR (s1.id = s2.id + 1
           AND s1.id = s3.id - 1)
       OR (s1.id = s2.id + 2
           AND s1.id = s3.id + 1))
ORDER BY id
```

# 620 有趣的电影
> 某城市开了一家新的电影院，吸引了很多人过来看电影。该电影院特别注意用户体验，专门有个 LED显示板做电影推荐，上面公布着影评和相关电影描述。
作为该电影院的信息部主管，您需要编写一个 SQL查询，找出所有影片描述为非 boring (不无聊) 的并且 id 为奇数 的影片，结果请按等级 rating 排列。
例如，下表 cinema:

![在这里插入图片描述](https://img-blog.csdnimg.cn/20200523000331332.png)

> 对于上面的例子，则正确的输出是为：

![在这里插入图片描述](https://img-blog.csdnimg.cn/20200523000350667.png)

**解法：** where 查询，order 排序

**注意：** 取余函数 %

```sql
SELECT *
FROM cinema
WHERE description != 'boring'
  AND id % 2 = 1
ORDER BY rating DESC
```

# 626 换座位
> 小美是一所中学的信息科技老师，她有一张 seat 座位表，平时用来储存学生名字和与他们相对应的座位 id。
> 其中纵列的 id 是连续递增的
> 小美想改变相邻俩学生的座位。
> 你能不能帮她写一个 SQL query 来输出小美想要的结果呢？
> 示例：

![在这里插入图片描述](https://img-blog.csdnimg.cn/20200523003329638.png)

> 假如数据输入的是上表，则输出结果如下：

![在这里插入图片描述](https://img-blog.csdnimg.cn/20200523003351290.png)

> 注意：
>
> - 如果学生人数是奇数，则不需要改变最后一个同学的座位。

**解法1：** 采用 case 进行调整

**注意：** 

- 对 id 进行调整而非对 student 进行调整，可以直接加减而不用再取值
- 注意从常量中取值的方法

```sql
SELECT (CASE
            WHEN id%2 = 1
                 AND id != tmp.counts THEN id + 1
            WHEN id%2 = 1
                 AND id = tmp.counts THEN id
            ELSE id -1
        END) AS id,
       student
FROM seat,

  (SELECT count(*) AS counts
   FROM seat) AS tmp
ORDER BY id
```


**解法2：** 采用自连接和位运算 (id + 1) ^ 1) - 1，之间将 1 -> 2，2 -> 1

```sql
SELECT
    s1.id, COALESCE(s2.student, s1.student) AS student
FROM
    seat s1
        LEFT JOIN
    seat s2 ON ((s1.id + 1) ^ 1) - 1 = s2.id
ORDER BY s1.id
```

# 627 交换工资
> 给定一个 salary 表，如下所示，有 m = 男性 和 f = 女性 的值。交换所有的 f 和 m 值（例如，将所有 f 值更改为 m，反之亦然）。要求只使用一个更新（Update）语句，并且没有中间的临时表。
注意，您必只能写一个 Update 语句，请不要编写任何 Select 语句。
例如：

![在这里插入图片描述](https://img-blog.csdnimg.cn/20200523001042709.png)

> 运行你所编写的更新语句之后，将会得到以下表：

![在这里插入图片描述](https://img-blog.csdnimg.cn/20200523001103930.png)

**解法：** update set 进行替换，采用 case when then else 进行交换

```sql
UPDATE salary
SET sex = CASE sex
              WHEN 'm' THEN 'f'
              ELSE 'm'
          END
```

#1179 
> 部门表 Department：

![在这里插入图片描述](https://img-blog.csdnimg.cn/20200523012109731.png)

> (id, month) 是表的联合主键。
这个表格有关于每个部门每月收入的信息。
月份（month）可以取下列值 ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]。
编写一个 SQL 查询来重新格式化表，使得新的表中有一个部门 id 列和一些对应 每个月 的收入（revenue）列。
查询结果格式如下面的示例所示：

![在这里插入图片描述](https://img-blog.csdnimg.cn/20200523012146216.png)

> 注意，结果表有 13 列 (1个部门 id 列 + 12个月份的收入列)。

**解法：** 采用 sum(case when) 进行行转列

```sql
SELECT id,
       sum(CASE `month` WHEN 'Jan' THEN revenue ELSE NULL END) AS Jan_Revenue,
       sum(CASE `month` WHEN 'Feb' THEN revenue ELSE NULL END) AS Feb_Revenue,
       sum(CASE `month` WHEN 'Mar' THEN revenue ELSE NULL END) AS Mar_Revenue,
       sum(CASE `month` WHEN 'Apr' THEN revenue ELSE NULL END) AS Apr_Revenue,
       sum(CASE `month` WHEN 'May' THEN revenue ELSE NULL END) AS May_Revenue,
       sum(CASE `month` WHEN 'Jun' THEN revenue ELSE NULL END) AS Jun_Revenue,
       sum(CASE `month` WHEN 'Jul' THEN revenue ELSE NULL END) AS Jul_Revenue,
       sum(CASE `month` WHEN 'Aug' THEN revenue ELSE NULL END) AS Aug_Revenue,
       sum(CASE `month` WHEN 'Sep' THEN revenue ELSE NULL END) AS Sep_Revenue,
       sum(CASE `month` WHEN 'Oct' THEN revenue ELSE NULL END) AS Oct_Revenue,
       sum(CASE `month` WHEN 'Nov' THEN revenue ELSE NULL END) AS Nov_Revenue,
       sum(CASE `month` WHEN 'Dec' THEN revenue ELSE NULL END) AS Dec_Revenue
FROM Department
GROUP BY id
```

