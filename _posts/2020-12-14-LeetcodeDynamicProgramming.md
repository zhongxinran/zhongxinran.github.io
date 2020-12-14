---
title: 力扣刷题 | 动态规划问题解法、例题及代码
author: 钟欣然
date: 2020-12-12 00:45:00 +0800
categories: [力扣刷题, 按问题分类]
math: true
mermaid: true
---

# 总述

动态规划（英语：Dynamic programming，简称 DP）是一种在数学、管理科学、计算机科学、经济学和生物信息学中使用的，通过把原问题分解为相对简单的子问题的方式求解复杂问题的方法。

动态规划常常适用于有重叠子问题和最优子结构性质的问题，动态规划方法所耗时间往往远少于朴素解法。

动态规划背后的基本思想非常简单。大致上，若要解一个给定问题，我们需要解其不同部分（即子问题），再根据子问题的解以得出原问题的解。动态规划往往用于优化递归问题，例如斐波那契数列，如果运用递归的方式来求解会重复计算很多相同的子问题，利用动态规划的思想可以减少计算量。

通常许多子问题非常相似，为此动态规划法试图仅仅解决每个子问题一次，具有天然剪枝的功能，从而减少计算量：一旦某个给定子问题的解已经算出，则将其记忆化存储，以便下次需要同一个子问题解之时直接查表。这种做法在重复子问题的数目关于输入的规模呈指数增长时特别有用。

# 总结
动态规划适用于有重复的子问题的情况，依次解决第一个到第n个子问题，和递归的区别是每个子问题只计算一次，后面直接调用结果

# 5 最长回文子串

> 给定一个字符串 s，找到 s 中最长的回文子串。你可以假设 s 的最大长度为 1000。

**解法：** 以每个或者每两个字符为中心，向两端拓展，最终取最长的。

```python
class Solution:
    def longestPalindrome(self, s: str) -> str:
        indexs = [[len(s)-1,len(s)-1]]
        for j in range(len(s)-1):
            indexs.append([j,j])
            if s[j] == s[j+1]:
                indexs.append([j,j+1])
        length = []
        for k in range(len(indexs)):
            while indexs[k][0] > 0 and indexs[k][1] < len(s)-1 and s[indexs[k][0]-1] == s[indexs[k][1]+1]:
                indexs[k] = [indexs[k][0]-1,indexs[k][1]+1]
            length.append(indexs[k][1]-indexs[k][0])
        index_final = indexs[length.index(max(length))]
        return s[index_final[0]:(index_final[1]+1)]
```

# 53 最大子序和
> 给定一个整数数组 nums ，找到一个具有最大和的连续子数组（子数组最少包含一个元素），返回其最大和。

**解法：** 从头开始，对每个数计算current_sum和sum_max，例如：

|-|-2|1|-3|4|-1|2|1|-5|4|
|-|-|-|-|-|-|-|-|-|-|
|current_sum|-2|1|-2|4|3|5|6|1|5|
|sum_max|-2|1|1|4|4|5|6|6|6|

```
if 当前数 + 上一个current_sum > 当前数:
	current_sum = 当前数 + current_sum
else:
	current_sum = 当前数

if current_sum > sum_max:
	sum_max = current_sum
else:
	sum_max = sum_max
```

```python
class Solution:
    def cross_sum(self,left,right):
        n_left,n_right = len(left),len(right)
        leftsum_max,rightsum_max = left[n_left-1],right[0]
        leftcurrent_sum,rightcurrent_sum = left[n_left-1],right[0]
        if n_left > 1:
            for i in range(n_left-1):
                leftcurrent_sum = leftcurrent_sum + left[n_left-2-i]
                leftsum_max = max(leftsum_max,leftcurrent_sum)
        if n_right > 1:
            for i in range(1,n_right):
                rightcurrent_sum = rightcurrent_sum + right[i]
                rightsum_max = max(rightsum_max,rightcurrent_sum)
        return (leftsum_max + rightsum_max)
    def maxSubArray(self, nums: List[int]):
        n = len(nums)
        if n == 1:
            return nums[0]
        left = nums[0:math.ceil(n/2)]
        right = nums[math.ceil(n/2):n]
        return max(self.maxSubArray(left),self.maxSubArray(right),self.cross_sum(left,right))

    '''
    # 动态规划
    def maxSubArray(self, nums: List[int]) -> int:
        current_max,sum_max = nums[0],nums[0]
        for num in nums[1:]:
            current_max = (current_max + num) if current_max > 0 else num
            sum_max = current_max if current_max > sum_max else sum_max
        return sum_max
    '''
```

# 62 不同路径
> 一个机器人位于一个 m x n 网格的左上角。机器人每次只能向下或者向右移动一步。机器人试图达到网格的右下角。问总共有多少条不同的路径？

**解法：** m×n路径数 = (m-1)×n路径数 + m×(n-1)路径数。
**重点：** 存储已经计算过的值避免重复计算
```python
class Solution:
    def uniquePaths(self, m: int, n: int) -> int:
        dp = [[1]*n for _ in range(m)]
        for i in range(1,m):
            for j in range(1,n):
                dp[i][j] = dp[i-1][j] + dp[i][j-1]
        return dp[m-1][n-1]
        '''
        # 超出时间限制
        if m == 1 or n == 1:
            return 1
        else:
            return self.uniquePaths(m-1,n)+self.uniquePaths(m,n-1)
        '''
```

# 63 不同路径2
> 一个机器人位于一个 m x n 网格的左上角。机器人每次只能向下或者向右移动一步。机器人试图达到网格的右下角。现在考虑网格中有障碍物，网格中的障碍物和空位置分别用 1 和 0 来表示。那么从左上角到右下角将会有多少条不同的路径？

**解法：** 在上题基础上，有障碍物则令当前点路径数为0，注意边缘行和列有障碍物会导致这一行或一列后面的点的路径数也为0

```python
class Solution:
    def uniquePathsWithObstacles(self, obstacleGrid: List[List[int]]) -> int:
        m = len(obstacleGrid)
        n = len(obstacleGrid[0])
        dp = [[1]*n for _ in range(m)]
        for i in range(m):
            for j in range(n):
                if obstacleGrid[i][j] == 1:
                    dp[i][j] = 0
                elif i == 0 and dp[i][j-1] == 0:
                    dp[i][j] = 0
                elif j == 0 and dp[i-1][j] == 0:
                    dp[i][j] = 0
                elif i != 0 and j != 0:
                    dp[i][j] = dp[i-1][j] + dp[i][j-1]
        return dp[m-1][n-1]
```

# 64 最小路径和
> 给定一个包含非负整数的 m x n 网格，请找出一条从左上角到右下角的路径，使得路径上的数字总和为最小。每次只能向下或者向右移动一步。

**解法：** (m,n)位置数字和 = (m,n)位置数字 + min{(m-1,n) 位置数字和,(m,n-1)位置数字和}

```python
class Solution:
    def minPathSum(self, grid: List[List[int]]) -> int:
        m,n = len(grid),len(grid[0])
        dp = grid
        for i in range(1,m):
            dp[i][0] = dp[i-1][0] + grid[i][0]
        for j in range(1,n):
            dp[0][j] = dp[0][j-1] + grid[0][j]
        for i in range(1,m):
            for j in range(1,n):
                dp[i][j] = grid[i][j] + min(dp[i][j-1],dp[i-1][j])
        return dp[m-1][n-1]
```

# 70 爬楼梯
> 假设你正在爬楼梯。需要 n 阶你才能到达楼顶。每次你可以爬 1 或 2 个台阶。你有多少种不同的方法可以爬到楼顶呢？注意：给定 n 是一个正整数。

**解法：** 每次走1步或两步，动态规划

```python
class Solution:
    def climbStairs(self, n: int) -> int:
        if n == 1:
            return 1
        elif n == 2:
            return 2
        dp = [1,2]
        for i in range(2,n):
            dp.append(dp[i-1]+dp[i-2])
        return dp[n-1]
```
# 91 编码方法
> 一条包含字母 A-Z 的消息通过以下方式进行了编码：
'A' -> 1
'B' -> 2
...
'Z' -> 26
给定一个只包含数字的非空字符串，请计算解码方法的总数。

**解法：** 从后往前依次算，有0的时候特殊处理

```python
class Solution:
    def numDecodings(self, s: str) -> int:
        n = len(s)
        if n == 1:
            if s[0] == '0':
                return 0
            else:
                return 1
        if s[n-1] == '0':
            if s[n-2] == '1' or s[n-2] == '2':
                dp = [0,1]
            else: return 0
        elif s[n-1] in '123456':
            if s[n-2] == '0':
                if n>2 and s[n-3] in '12':
                    dp = [1,0]
                else:
                    return 0
            elif s[n-2] in '12':
                dp = [1,2]
            else:
                dp = [1,1]
        elif s[n-1] in '789':
            if s[n-2] == '0':
                if n>2 and s[n-3] in '12':
                    dp = [1,0]
                else:
                    return 0
            elif s[n-2] in '1':
                dp = [1,2]
            else:
                dp = [1,1]
        for i in range(2,n):
            j = n-1-i
            if s[j] == '0':
                if j == 0 or (s[j-1] != '1' and s[j-1] != '2'):
                    return 0
                else:
                    dp.append(0)
            elif s[j] == '1':
                if s[j+1] == '0':
                    dp.append(dp[i-2])
                else:
                    dp.append(dp[i-1]+dp[i-2])
            elif s[j] == '2':
                if s[j+1] == '0':
                    dp.append(dp[i-2])
                elif s[j+1] in '123456':
                    dp.append(dp[i-1]+dp[i-2])
                else:
                    dp.append(dp[i-1])
            else:
                if s[j+1] == '0':
                    return 0
                else:
                    dp.append(dp[i-1])
        return dp[n-1]
```

# 95 不同的二叉搜索树2
> 给定一个整数 n，生成所有由 1 ... n 为节点所组成的二叉搜索树。

**解法：** 将1-n每个数字分别作为根节点，左子树为小于这个节点的数，右子树为大于这个节点的数，递归
```python
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution:
    def generateTrees(self, n: int) -> List[TreeNode]:
        def generate_trees(start, stop):
            if start > stop:
                return [None]
            elif start == stop:
                return [TreeNode(start)]
            dp = []
            for i in range(start,stop+1):
                left_trees = generate_trees(start,i-1)
                right_trees = generate_trees(i+1,stop)
                for l in left_trees:
                    for r in right_trees:
                        temp = TreeNode(i)
                        temp.left = l
                        temp.right = r
                        dp.append(temp)
            return dp
        return generate_trees(1,n) if n else []
```
# 96 不同的二叉搜索树
> 给定一个整数 n，求以 1 ... n 为节点组成的二叉搜索树有多少种？

**解法：** 依次计算1、2、……、n个节点的二叉树有多少种，i个节点的二叉树种类为对左子树包含1到i-1个节点，对应的右子树包含i-1到1个节点，求左子树种类*右子树种类的和

```python
class Solution:
    def numTrees(self, n: int) -> int:
        dp = [1,1]
        if n >= 2:
            for i in range(2,n+1):
                temp = 0
                for j in range(i):
                    temp = temp + dp[j]*dp[i-j-1]
                dp.append(temp)
        return dp[n]
```
# 120 三角形最小路径和
> 给定一个三角形，找出自顶向下的最小路径和。每一步只能移动到下一行中相邻的结点上。

![在这里插入图片描述](https://img-blog.csdnimg.cn/20200104190239436.png)

**解法：** 从上到下对每个点找到当前点的最小路径，取最后一行的最小值

```python
class Solution:
    def minimumTotal(self, triangle: List[List[int]]) -> int:
        m = len(triangle)
        for i in range(1,m):
            for j in range(i+1):
                if j == 0:
                    triangle[i][j] = triangle[i][j] + triangle[i-1][j]
                elif j == i:
                    triangle[i][j] = triangle[i][j] + triangle[i-1][j-1]
                else:
                    triangle[i][j] = triangle[i][j] + min(triangle[i-1][j],triangle[i-1][j-1])
        return min(triangle[m-1])
```

# 121 买卖股票的最佳时机
> 给定一个数组，它的第 i 个元素是一支给定股票第 i 天的价格。如果你最多只允许完成一笔交易（即买入和卖出一支股票），设计一个算法来计算你所能获取的最大利润。注意你不能在买入股票前卖出股票。

**解法：** 记录每个时刻及该时刻以前价格的最低值，用每个时刻价格减去对应的价格最低值，对其和0取大

```python
class Solution:
    def maxProfit(self, prices: List[int]) -> int:
        if len(prices) == 0:
            return 0
        min_price = max(prices)
        probit = 0
        for i in range(len(prices)):
            min_price = min_price if prices[i] > min_price else prices[i]
            probit = (prices[i]-min_price) if (prices[i]-min_price) > probit else probit
        return probit if probit > 0 else 0
```

# 122 买卖股票的最佳时机2
> 给定一个数组，它的第 i 个元素是一支给定股票第 i 天的价格。设计一个算法来计算你所能获取的最大利润。你可以尽可能地完成更多的交易（多次买卖一支股票）。注意：你不能同时参与多笔交易（你必须在再次购买前出售掉之前的股票）。

**解法：** 第一天的利润为0，之后每天的利润为 前一天的利润 加上 当天价格和前一天价格的差和0取大

```python
class Solution:
    def maxProfit(self, prices: List[int]) -> int:
        dp = [0]
        for i in range(1,len(prices)):
            dp.append(dp[i-1]+max(prices[i]-prices[i-1],0))
        return dp[len(prices)-1]
```

# 123 买卖股票的最佳时机3
> 给定一个数组，它的第 i 个元素是一支给定的股票在第 i 天的价格。设计一个算法来计算你所能获取的最大利润。你最多可以完成 两笔 交易。注意: 你不能同时参与多笔交易（你必须在再次购买前出售掉之前的股票）。

**解法：** 用$dp[i][j][s]$表示每个时刻的状态的利润，其中i表示天数，j表示当前已经进行的交易次数（买入算交易），s表示当前时刻是否持有股票，则对第一天有

| |j=0|j=1|j=2|…|
|-|-|-|-|-|
|s=0|0|$-\infty$|$-\infty$|$-\infty$
|s=1|$-\infty$|-prices[0]|$-\infty$|$-\infty$|

对之后的每一天有

```python
# 当前时刻未持有股票可能是前一天也未持有股票或者前一天持有股票当天卖出（收获利润为当天价格）
dp[i][j][0] = max(dp[i-1][j][0],dp[i-1][j][1]+prices[i])
# 当前时刻持有股票可能是前一天也持有股票或者前一天未持有股票当天买入（收获利润为负的当天价格），j=0表示未进行交易不可能持有股票故利润为负无穷
dp[i][j][1] = max(dp[i-1][j][1],dp[i-1][j-1][0]-prices[i]) if j != 0 else -float('inf')
```

最终返回最后一天不同交易次数下未持有股票的利润最大值，即$max([dp[n-1][j][0]\;for\;j\;in\;range(k+1)])$，因为持有股票未卖出肯定不是最大利润

TIPS：可以用-float('inf')表示负无穷参与计算

```python
class Solution:
    def maxProfit(self, prices: List[int]):
        n = len(prices)
        k = 2
        if k == 0 or len(prices) <= 1:
            return 0
        dp = [[[0]*2 for _ in range(k+1)] for _ in range(n)]
        for j in range(k+1):
            for s in [0,1]:
                dp[0][j][s] = -float('inf')
        dp[0][0][0] = 0
        dp[0][1][1] = -prices[0]
        for i in range(1,n):
            for j in range(k+1):
                dp[i][j][0] = max(dp[i-1][j][0],dp[i-1][j][1]+prices[i])
                dp[i][j][1] = max(dp[i-1][j][1],dp[i-1][j-1][0]-prices[i]) if j != 0 else -float('inf')
        return max([dp[n-1][j][0] for j in range(k+1)])
    '''
    # 交易两次意味着在中间任意一点断开分别交易一次
    def one_maxProfit(self, prices: List[int]) -> int:
        if len(prices) == 0:
            return 0
        min_price = max(prices)
        probit = 0
        for i in range(len(prices)):
            min_price = min_price if prices[i] > min_price else prices[i]
            probit = (prices[i]-min_price) if (prices[i]-min_price) > probit else probit
        return probit if probit > 0 else 0
    def maxProfit(self, prices: List[int]) -> int:
        if len(prices) < 4:
            return self.one_maxProfit(prices)
        two_probit = []
        for i in range(2,len(prices)-1):
            if prices[i-1] >= prices[i] and prices[i] < prices[i+1]:
                two_probit.append(self.one_maxProfit(prices[0:i])+self.one_maxProfit(prices[i:]))
        return max(max(two_probit) if len(two_probit) > 0 else 0,self.one_maxProfit(prices))
    '''
```

# 139 单词拆分
> 给定一个非空字符串 s 和一个包含非空单词列表的字典 wordDict，判定 s 是否可以被空格拆分为一个或多个在字典中出现的单词。
说明：拆分时可以重复使用字典中的单词。你可以假设字典中没有重复的单词。

**解法：** 从前往后一个一个找，记录当前位置之前能不能拆分，方便后面调用

```python
class Solution:
    def wordBreak(self, s: str, wordDict: List[str]) -> bool:
        if len(wordDict) == 0:
            return False
        n = max([len(word) for word in wordDict])
        dp = [True] + [False]*len(s)
        for i in range(1,len(s)+1):
            for j in range(1,min(i,n)+1):
                if dp[i-j] and s[(i-j):i] in wordDict:
                    dp[i] = True
        return dp[len(s)]

        '''
        for i in range(n):
            temp = []
            if s[0:(i+1)] in wordDict:
                if len(s) == i+1:
                    return True
                else:
                    temp.append(self.wordBreak(s[(i+1):],wordDict))
            else:
                temp.append(False)
        if True in temp:
            return True
        else:
            return False
        '''
```

