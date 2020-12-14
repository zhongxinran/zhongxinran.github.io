---
title: 力扣刷题 | 栈问题解法、例题及代码
author: 钟欣然
date: 2020-12-12 00:48:00 +0800
categories: [力扣刷题, 按问题分类]
math: true
mermaid: true
---

# 总述
栈（Stack）又名堆栈，它是一种重要的数据结构。从数据结构角度看，栈也是线性表，其特殊性在于栈的基本操作是线性表操作的子集，它是操作受限的线性表，因此，可称为限定性的数据结构。限定它仅在表尾进行插入或删除操作。表尾称为栈顶，相应地，表头称为栈底。栈的基本操作除了在栈顶进行插入和删除外，还有栈的初始化，判空以及取栈顶元素等。

# 20 有效的括号
> 给定一个只包括 '('，')'，'{'，'}'，'['，']' 的字符串，判断字符串是否有效。
有效字符串需满足：
> - 左括号必须用相同类型的右括号闭合。
> - 左括号必须以正确的顺序闭合。
> - 注意空字符串可被认为是有效字符串。

**解法：** 采用栈的方式，注意其中几个小技巧，如快速判断奇数长度，对列表直接进行判断，检查其是否为空等

```python
class Solution:
    def isValid(self, s: str) -> bool:

        # 快速判断奇数长度
        if len(s) % 2 == 1:
            return False

        # 存储为哈希映射，便于读取
        pairs = {')': '(', '}': '{', ']': '['}

        # 采用栈的方式，快速判断
        temp = list()
        for i in s:
            if i in pairs:
                # 列表直接判断，空时为false，否则人为true
                if temp and temp[-1] == pairs[i]:
                    temp.pop()
                else:
                    return False
            else:
                temp.append(i)

        # 替代先 if 判断列表长度，再返回值
        return not temp
```

# 42 接雨水
> 给定 n 个非负整数表示每个宽度为 1 的柱子的高度图，计算按此排列的柱子，下雨之后能接多少雨水。
> 下图是由数组 [0,1,0,2,1,0,1,3,2,1,2,1] 表示的高度图，在这种情况下，可以接 6 个单位的雨水（蓝色部分表示雨水）。

![在这里插入图片描述](https://img-blog.csdnimg.cn/20200901214345405.png)

**解法1：** 暴力解法，首先去掉列表开头结尾的0，对中间的每个数，找到它前面的最大值和后面的最大值，将所在位置上的数与前后的最大值比较，判断此处能接的水数，时间复杂度$O(n^2)$



```python
class Solution:
    def trap(self, height: List[int]) -> int:
        # 判断列表长度，避免列表为空时报错
        while height and height[0] == 0:
            del height[0]
        while height and height[-1] == 0:
            del height[-1]

        if len(height) <= 2:
            return 0

        volumn = 0
        for i in range(1, len(height) - 1):
            temp = min(max(height[:i]), max(height[(i + 1):]))
            if height[i] < temp:
                volumn += temp - height[i]

        return volumn
```

**解法2：** 单调递减栈，存储 height 列表的索引，从左至右遍历，每次出现比当前栈顶对应的 height 元素更高的柱子，就结算一次，维护栈是单调递减的（即依次结算栈中元素，直到当前栈顶对应的 height 元素比现在的柱子更高）

```python
class Solution:
    def trap(self, height: List[int]) -> int:
        if len(height) <= 2:
            return 0

        stack = list()
        volumn = 0
        for i in range(len(height)):
            while stack and height[stack[-1]] < height[i]:
                temp = height[stack[-1]]
                del stack[-1]
                if stack:
                    volumn += (min(height[stack[-1]], height[i]) -
                               temp) * (i - stack[-1] - 1)
            stack.append(i)
        return volumn
```

# 71 简化路径

> 以 Unix 风格给出一个文件的绝对路径，你需要简化它。或者换句话说，将其转换为规范路径。
> 在 Unix 风格的文件系统中，一个点（.）表示当前目录本身；此外，两个点 （..） 表示将目录切换到上一级（指向父目录）；两者都可以是复杂相对路径的组成部分。更多信息请参阅：Linux / Unix中的绝对路径 vs 相对路径
请注意，返回的规范路径必须始终以斜杠 / 开头，并且两个目录名之间必须只有一个斜杠 /。最后一个目录名（如果存在）不能以 / 结尾。此外，规范路径必须是表示绝对路径的最短字符串。

> 示例 1：
输入："/home/"
输出："/home"
解释：注意，最后一个目录名后面没有斜杠。

> 示例 2：
输入："/../"
输出："/"
解释：从根目录向上一级是不可行的，因为根是你可以到达的最高级。

> 示例 3：
输入："/home//foo/"
输出："/home/foo"
解释：在规范路径中，多个连续斜杠需要用一个斜杠替换。

> 示例 4：
输入："/a/./b/../../c/"
输出："/c"

> 示例 5：
输入："/a/../../b/../c//.//"
输出："/c"

> 示例 6：
输入："/a//b////c/d//././/.."
输出："/a/b/c"


**解法：** 采用栈维护各级目录，遇到 . 不变，遇到 .. 删除栈顶，遇到 / 跳过，其它情况向栈中添加元素，注意， $.$、$..$、目录名的判断均要以 / 结尾或到达字符串尾部

```python
class Solution:
    def simplifyPath(self, path: str) -> str:
        stack = list()
        i = 0

        while i < len(path):
            if path[i] == '.'and ((i+1) >= len(path) or path[i+1] == '/'):
                i += 1
                continue
            if path[i] == '.'and (i+1) < len(path) and path[i+1] == '.' and ((i+2) >= len(path) or path[i+2] == '/'): 
                if stack:
                    del stack[-1]
                i += 2
                continue

            if path[i] != '/':
                j = 1
                while i+j < len(path) and path[i+j] != '/':
                    j += 1
                stack.append(path[i:(i+j)])
                i += j
            else:
                i += 1

        return '/'+'/'.join(stack)
```
# 84 柱状图中的最大矩形
> 给定 n 个非负整数，用来表示柱状图中各个柱子的高度。每个柱子彼此相邻，且宽度为 1 。求在该柱状图中，能够勾勒出来的矩形的最大面积。
> 以下是柱状图的示例，其中每个柱子的宽度为 1，给定的高度为 [2,1,5,6,2,3]。图中阴影部分为所能勾勒出的最大矩形面积，其面积为 10 个单位。

![在这里插入图片描述](https://img-blog.csdnimg.cn/20200902213032247.png)

**解法1：** 暴力解法，时间复杂度$O(n^2)$，空间复杂度$O(1)$超出时间限制，对每个柱子分别找到左边和右边比这个柱子矮的第一根柱子，分别求矩形的高和宽

```python
class Solution:
    def largestRectangleArea(self, heights: List[int]) -> int:
        if len(heights) == 0:
            return 0
        elif len(heights) == 1:
            return heights[0]

        ans = 0
        for i in range(len(heights)):
            j = i+1
            k = i-1
            while j < len(heights) and heights[j] >= heights[i]:
                j += 1
            while k >= 0 and heights[k] >= heights[i]:
                k -= 1
            ans = max(ans, (j-k-1)*heights[i])

        return ans
```
**解法2：** 单调栈，时间复杂度$O(n)$，空间复杂度$O(n)$
* 首先从左到右执行，维护一个单调递增栈，顺便记录每个柱子左边比它矮的第一个柱子的索引，如左边没有更矮的，则记为-1
* 其次从右到左执行，同样维护一个单调递增栈，顺便记录每个柱子右边比它矮的第一个柱子的索引，如右边没有更矮的，则记为列表长度
* 对每个柱子，利用上述记录的左右两边的索引和柱子高度，求出矩形面积，再求最大值

```python
class Solution:
    def largestRectangleArea(self, heights: List[int]) -> int:
        if len(heights) == 0:
            return 0
        elif len(heights) == 1:
            return heights[0]

        bin_left_index = list()
        bin_left_stack = list()
        for i in range(len(heights)):
            while bin_left_stack and heights[i] <= heights[bin_left_stack[-1]]:
                del bin_left_stack[-1]
            bin_left_index.append(bin_left_stack[-1]) if bin_left_stack else bin_left_index.append(-1)
            bin_left_stack.append(i)
        
        bin_right_index = list()
        bin_right_stack = list()
        for j in range(len(heights)):
            while bin_right_stack and heights[len(heights)-j-1] <= heights[bin_right_stack[-1]]:
                del bin_right_stack[-1]
            bin_right_index.append(bin_right_stack[-1]) if bin_right_stack else bin_right_index.append(len(heights))
            bin_right_stack.append(len(heights)-j-1)

        ans = 0
        for i in range(len(heights)):
            ans = max(ans, heights[i]*(bin_right_index[len(heights)-i-1]-bin_left_index[i]-1))
        return ans
```

# 85 最大矩形
> 给定一个仅包含 0 和 1 的二维二进制矩阵，找出只包含 1 的最大矩形，并返回其面积。

> 示例：
输入：
[
  ["1","0","1","0","0"],
  ["1","0","1","1","1"],
  ["1","1","1","1","1"],
  ["1","0","0","1","0"]
]
输出： 6

**解法：** 在上题基础上，逐行计算。用一个长度为矩阵宽度的列表存储每行的柱状图数据，列表中的每个元素（即柱子高度）为该元素所在列中本行及以上行连续的 1 的个数，如示例数据中，第一行应为 1 0 1 0 1，第二行应为 2 0 2 1 1，第三行应为3 1 3 2 2，第四行应为 4 0 0 3 0，用 84 题的方法计算每行柱状图中最大矩形的面积，取各行的最大值，即为答案

```python
class Solution:
    def largestRectangleArea(self, heights: List[int]) -> int:
        if len(heights) == 0:
            return 0
        elif len(heights) == 1:
            return heights[0]

        bin_left_index = list()
        bin_left_stack = list()
        for i in range(len(heights)):
            while bin_left_stack and heights[i] <= heights[bin_left_stack[-1]]:
                del bin_left_stack[-1]
            bin_left_index.append(bin_left_stack[-1]) if bin_left_stack else bin_left_index.append(-1)
            bin_left_stack.append(i)
        
        bin_right_index = list()
        bin_right_stack = list()
        for j in range(len(heights)):
            while bin_right_stack and heights[len(heights)-j-1] <= heights[bin_right_stack[-1]]:
                del bin_right_stack[-1]
            bin_right_index.append(bin_right_stack[-1]) if bin_right_stack else bin_right_index.append(len(heights))
            bin_right_stack.append(len(heights)-j-1)

        ans = 0
        for i in range(len(heights)):
            ans = max(ans, heights[i]*(bin_right_index[len(heights)-i-1]-bin_left_index[i]-1))
        return ans


    def maximalRectangle(self, matrix: List[List[str]]) -> int:
        if not matrix:
            return 0
        
        ans = 0
        accu_by_column = [0] * len(matrix[0])
        for i in range(len(matrix)):
            for j in range(len(matrix[0])):
                accu_by_column[j] = accu_by_column[j] + 1 if matrix[i][j] == '1' else 0
            ans = max(ans, self.largestRectangleArea(accu_by_column))
        return ans
```

# 94 二叉树的中序遍历
> 给定一个二叉树，返回它的中序 遍历。

> 示例:

![](https://img-blog.csdnimg.cn/20200902224145538.png)

**解法：** 栈，利用栈保存未处理完的树，每次遇到左子树为空，弹出一棵树，记录其根的值，并处理其右子树

```python
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution:
    def inorderTraversal(self, root: TreeNode) -> List[int]:
        if not root:
            return []
        
        stack = list()
        res = list()
        while root or stack:
            if root:
                stack.append(root)
                root = root.left
            else:
                root = stack.pop(-1)
                res.append(root.val)
                root = root.right

        return res
```

