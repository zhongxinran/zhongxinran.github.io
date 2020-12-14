---
title: 力扣刷题 | 贪心问题解法、例题及代码
author: 钟欣然
date: 2020-12-12 00:50:00 +0800
categories: [力扣刷题, 按问题分类]
math: true
mermaid: true
---

# 总述
贪心算法（又称贪婪算法）是指，在对问题求解时，总是做出在当前看来是最好的选择。也就是说，不从整体最优上加以考虑，他所做出的是在某种意义上的局部最优解。

贪心算法不是对所有问题都能得到整体最优解，关键是贪心策略的选择，选择的贪心策略必须具备无后效性，即某个状态以前的过程不会影响以后的状态，只与当前状态有关。

# 44 通配符匹配
> 给定一个字符串 (s) 和一个字符模式 (p) ，实现一个支持 '?' 和 '*' 的通配符匹配。两个字符串完全匹配才算匹配成功。
> - '?' 可以匹配任何单个字符。
> - '*' 可以匹配任意字符串（包括空字符串）。

> 说明:
> - s 可能为空，且只包含从 a-z 的小写字母。
> - p 可能为空，且只包含从 a-z 的小写字母，以及字符 ? 和 *。

**解法1：** 双指针贪心算法，对 s 和 p 的字符逐一匹配，匹配的位置分为记为 i, j，分为如下几种情况

- 如果 s[i] 和 p[j] 可以匹配（包括完全一致和 p[j] 为 ?），则 i, j 都前进一格
- 否则，如果 p[j] 为 *，将此时的 i, j 的位置记录为 i_star, j_star，j 前进一格，i 不动（因为 * 可以匹配空字符串，所以此处先不匹配，先往后看，后续不能匹配时再回溯 i, j 到 i_star+1, j_star+1）
- 否则（注意，此处的否则指不满足以上两种情况，即既不能匹配，p[j] 也不为 *，此时如果前面曾经出现过 *，可以回退到 * 位置），如果 * 出现过，回溯 i, j 到 i_star+1, j_star+1，i_star 也前进一格
- 否则（不能匹配，p[j] 也不为 *，前面未出现过 *），结束检查，返回 false

其它值得注意的地方：

- 应把 s 匹配完为止，否则如果以匹配完 p 为止，可能出现 "aaaa", "***a" 时 s 未匹配完，p 已经匹配完，导致错误结果
- 匹配结束后，应将 p 结尾处尚未参与匹配的 * 删掉（如有），如果此时 p 也匹配完了，则返回 true，否则返回false

```python
class Solution:
    def isMatch(self, s: str, p: str) -> bool:
        if len(p) == 0:
            return False if len(s) > 0 else True
        
        i, j, i_star, j_star = 0, 0, -1, -1
        while i < len(s):
            print(i,j)
            if j < len(p) and (s[i] == p[j] or p[j] == '?'):
                i += 1
                j += 1
            elif j < len(p) and p[j] == '*':
                i_star, j_star = i, j
                j += 1
            elif i_star >= 0:
                if j == len(p) and p[-1] == '*':
                    return True
                j = j_star + 1
                i_star += 1
                i = i_star
            else:
                return False

        while j < len(p) and p[j] == '*':
            j += 1
        return j == len(p)
```

**解法2：** 动态规划，dp[i][j] 表示 p 的前 j 个字符是否和 s 的前 j 个字符匹配，状态转移方程为

$$
dp\lbrack i\rbrack\lbrack j\rbrack\;=\;\left\{\begin{array}{l}dp\lbrack i-1\rbrack\lbrack j-1\rbrack\;and\;(p\lbrack j-1\rbrack\;==\;s\lbrack i-1\rbrack\;or\;p\lbrack j-1\rbrack\;==\;"?"),\;p\lbrack j-1\rbrack\;!=\;'\ast'\\dp\lbrack i-1\rbrack\lbrack j\rbrack\;or\;dp\lbrack i\rbrack\lbrack j-1\rbrack,\;p\lbrack j-1\rbrack\;==\;'\ast'\end{array}\right.
$$

- $p[j-1]$不为 * 时，要看 p 的前 j-1 个字符能否匹配 s 的前 i-1 个字符以及 $p[j-1]$ 和 $s[i-1]$是否匹配，此处的匹配包含问号和非问号两种情况
- $p[j-1]$为 * 时，分为 * 是否使用两种情况，使用时看 p 的前 j 个字符能否匹配 s 的前 i-1 个字符，不使用时看 p 的前 j-1 个字符能否匹配 s 的前 i 个字符，二者为或的关系

边界情况为：
- $dp[0][0] = true$，二者都为空时，可以匹配
- $dp[i][0] = false, i>1$，p 为空 s 不为空时，不能匹配
- $dp[0][j] = dp[0][j-1]\;and\;p[j-1] == "*", j>1$，s 为空 p 不为空时，要看 p 是否全部为 *

```python
class Solution:
    def isMatch(self, s: str, p: str) -> bool:
        if len(p) == 0:
            return False if len(s) > 0 else True
        
        dp = [[False]*(len(p)+1) for _ in range(len(s)+1)]
        dp[0][0] = True
        for j in range(1, len(p)+1):
            dp[0][j] = dp[0][j-1] and p[j-1] == '*'
        for i in range(1, len(s)+1):
            for j in range(1, len(p)+1):
                if p[j-1] != '*':
                    dp[i][j] = dp[i-1][j-1] and (s[i-1] == p[j-1] or p[j-1] == '?')
                else:
                    dp[i][j] = dp[i][j-1] or dp[i-1][j]
        return dp[len(s)][len(p)]
```

# 55 跳跃游戏
> 给定一个非负整数数组，你最初位于数组的第一个位置。数组中的每个元素代表你在该位置可以跳跃的最大长度。判断你是否能够到达最后一个位置。

> 示例 1:
- 输入: [2,3,1,1,4]；
- 输出: true；
- 解释: 我们可以先跳 1 步，从位置 0 到达 位置 1, 然后再从位置 1 跳 3 步到达最后一个位置。

> 示例 2:
- 输入: [3,2,1,0,4]；
- 输出: false；
- 解释: 无论怎样，你总会到达索引为 3 的位置。但该位置的最大跳跃长度是 0 ， 所以你永远不可能到达最后一个位置。

**解法1：** 贪心算法，遍历整个数组，计算每个位置（如本身可达）的最远可达位置，始终维护最远可达位置，遍历结束后查看最远可达位置是否大于数组长度

```python
class Solution:
    def canJump(self, nums: List[int]) -> bool:
        if len(nums) <= 1:
            return True
        
        max_loc = 0
        for i in range(len(nums)):
            if i <= max_loc:
                max_loc = max(max_loc, i + nums[i])
        
        return max_loc >= len(nums)-1
```

**解法2：** 贪心算法，在每个位置上跳跃最大长度，维护最远可达距离及到达此位置的次数，根据到达此位置的次数回退，最远可达距离大于数组长度或回退到0位置时停止，思路较复杂，不推荐采用

```python
class Solution:
    def canJump(self, nums: List[int]) -> bool:
        if len(nums) <= 1:
            return True
        if nums[0] == 0:
            return False 
        
        i, j = nums[0], nums[0]
        num = 1
        while i > 0 and j < len(nums)-1:
            print(i,j,num)
            if nums[i] > 0:
                i += nums[i]
                if i > j:
                    j = i
                    num = 1
                elif i == j:
                    num += 1
            else:
                while i-num+nums[i-num] <= j and num < j:
                    num += 1
                i -= num
        return j >= len(nums)-1
```

# 45 跳跃游戏 2
> 给定一个非负整数数组，你最初位于数组的第一个位置。数组中的每个元素代表你在该位置可以跳跃的最大长度。你的目标是使用最少的跳跃次数到达数组的最后一个位置。假设你总是可以到达数组的最后一个位置。

> 示例:
- 输入: [2,3,1,1,4]；
- 输出: 2；
- 解释: 跳到最后一个位置的最小跳跃数是 2。从下标为 0 跳到下标为 1 的位置，跳 1 步，然后跳 3 步到达数组的最后一个位置。


**解法：** 计算每一格能达到的最远位置，从第一个出发，跳到当前可达的最远位置，视为第一步，取从零位置到当前可达的最远位置之间每个格子可达位置中最大的一个，跳过去，视为第二步，重复第二步，直到到达终点

```python
class Solution:
    def jump(self, nums: List[int]) -> int:
        if len(nums) <= 1:
            return 0
        
        loc = len(nums)-1
        max_loc = [i + nums[i] for i in range(len(nums))]

        loc = nums[0]
        i = 1
        while loc < len(nums)-1:
            loc = max(max_loc[:(loc+1)])
            i += 1
        return i
```
**代码优化：**

```python
class Solution:
    def jump(self, nums: List[int]) -> int:
        n = len(nums)
        maxPos, end, step = 0, 0, 0
        for i in range(n - 1):
            if maxPos >= i:
                maxPos = max(maxPos, i + nums[i])
                if i == end:
                    end = maxPos
                    step += 1
        return step
```

# 122 买卖股票的最佳时机 2
> 给定一个数组，它的第 i 个元素是一支给定股票第 i 天的价格。设计一个算法来计算你所能获取的最大利润。你可以尽可能地完成更多的交易（多次买卖一支股票）。
注意：你不能同时参与多笔交易（你必须在再次购买前出售掉之前的股票）。

 > 示例 1:
- 输入: [7,1,5,3,6,4]；
- 输出: 7；
- 解释: 在第 2 天（股票价格 = 1）的时候买入，在第 3 天（股票价格 = 5）的时候卖出, 这笔交易所能获得利润 = 5-1 = 4 。随后，在第 4 天（股票价格 = 3）的时候买入，在第 5 天（股票价格 = 6）的时候卖出, 这笔交易所能获得利润 = 6-3 = 3 。

> 示例 2:
- 输入: [1,2,3,4,5]；
- 输出: 4；
- 解释: 在第 1 天（股票价格 = 1）的时候买入，在第 5 天 （股票价格 = 5）的时候卖出, 这笔交易所能获得利润 = 5-1 = 4 。注意你不能在第 1 天和第 2 天接连购买股票，之后再将它们卖出。因为这样属于同时参与了多笔交易，你必须在再次购买前出售掉之前的股票。

> 示例 3:
- - 输入: [7,6,4,3,1]；
- 输出: 0；
- 解释: 在这种情况下, 没有交易完成, 所以最大利润为 0。

> 提示：
- 1 <= prices.length <= 3 * 10 ^ 4
- 0 <= prices[i] <= 10 ^ 4

**解法1：** 贪心算法的思路，动态规划的解法，当天价格比上一天高，就交易一次，比上一天低，就不交易

```python
class Solution:
    def maxProfit(self, prices: List[int]) -> int:
        dp = [0]
        for i in range(1,len(prices)):
            dp.append(dp[i-1]+max(prices[i]-prices[i-1],0))
        return dp[len(prices)-1]
```

# 134 加油站
> 在一条环路上有 N 个加油站，其中第 i 个加油站有汽油 gas[i] 升。你有一辆油箱容量无限的的汽车，从第 i 个加油站开往第 i+1 个加油站需要消耗汽油 cost[i] 升。你从其中的一个加油站出发，开始时油箱为空。如果你可以绕环路行驶一周，则返回出发时加油站的编号，否则返回 -1。

> 说明: 
> - 如果题目有解，该答案即为唯一答案。
> - 输入数组均为非空数组，且长度相同。
> - 输入数组中的元素均为非负数。

> 示例 1:
- 输入: gas  = [1,2,3,4,5]，- cost = [3,4,5,1,2]
- 输出: 3
- 解释:
从 3 号加油站(索引为 3 处)出发，可获得 4 升汽油。此时油箱有 = 0 + 4 = 4 升汽油；
开往 4 号加油站，此时油箱有 4 - 1 + 5 = 8 升汽油；
开往 0 号加油站，此时油箱有 8 - 2 + 1 = 7 升汽油；
开往 1 号加油站，此时油箱有 7 - 3 + 2 = 6 升汽油；
开往 2 号加油站，此时油箱有 6 - 4 + 3 = 5 升汽油；
开往 3 号加油站，你需要消耗 5 升汽油，正好足够你返回到 3 号加油站。
因此，3 可为起始索引。

> 示例 2:
- 输入: gas  = [2,3,4]，cost = [3,4,3]
- 输出: -1
- 解释:
你不能从 0 号或 1 号加油站出发，因为没有足够的汽油可以让你行驶到下一个加油站。
我们从 2 号加油站出发，可以获得 4 升汽油。 此时油箱有 = 0 + 4 = 4 升汽油；
开往 0 号加油站，此时油箱有 4 - 3 + 2 = 3 升汽油；
开往 1 号加油站，此时油箱有 3 - 3 + 3 = 3 升汽油；
你无法返回 2 号加油站，因为返程需要消耗 4 升汽油，但是你的油箱只有 3 升汽油。
因此，无论怎样，你都不可能绕环路行驶一周。

**解法1：** 暴力解法，遍历从每个加油站出发，检查中间油是否够用

```python
class Solution:
    def canCompleteCircuit(self, gas: List[int], cost: List[int]) -> int:
        i = 0
        while i < len(gas):
            volumn = 0
            for j in range(i, i+len(gas)):
                volumn = volumn + gas[j%len(gas)] - cost[j%len(gas)]
                if volumn < 0:
                    break
            if volumn >= 0:
                return i
            i += 1
        return -1
```

**解法2：** 首先计算总油量和总耗油的差，如果小于 0，返回 -1，大于 0 则一定有解。从 0 站开始遍历，汽车从 start 位置出发，逐一检查各站油量，如在 start’ 站前油第一次不够，则正确的出发站一定位于 start' 站及之后。原因如下：

- 如果正确的出发站位于 0-start 之间，则不会遍历至 start 作为出发站的情况
- 如果正确的出发站位于 start-start' 之间，假设为 start''，则必然有 start-start''、start''-start' 二者其一油不够，前者不符合 start 出发在 start' 第一次油不够的设定，后者则导致从 start'' 出发也不能到达 start，与start'' 是正确的出发站矛盾。

综上，可通过一次遍历，找到正确的出发站。

```python
class Solution:
    def canCompleteCircuit(self, gas: List[int], cost: List[int]) -> int:
        if sum(gas)-sum(cost) < 0:
            return -1
        
        start = 0
        total_gas = 0
        for i in range(len(gas)):
            if total_gas < 0:
                start = i
                total_gas = gas[i] - cost[i]
            else:
                total_gas = total_gas + gas[i] - cost[i]
        return start
```

