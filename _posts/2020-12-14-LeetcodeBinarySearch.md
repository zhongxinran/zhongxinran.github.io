---
title: 力扣刷题 | 二分查找问题解法、例题及代码
author: 钟欣然
date: 2020-12-12 00:47:00 +0800
categories: [力扣刷题, 按问题分类]
math: true
mermaid: true
---

# 总述
二分查找也称折半查找（Binary Search），它是一种效率较高的查找方法，前提是数据结构必须先排好序，可以在数据规模的对数时间复杂度内完成查找。但是，二分查找要求线性表具有有随机访问的特点（例如数组），也要求线性表能够根据中间元素的特点推测它两侧元素的性质，以达到缩减问题规模的效果。

# 4 寻找两个正序数组的中位数
> 给定两个大小为 m 和 n 的正序（从小到大）数组 nums1 和 nums2。请你找出这两个正序数组的中位数，并且要求算法的时间复杂度为 O(log(m + n))。
你可以假设 nums1 和 nums2 不会同时为空。

**解法：** 对两个数组进行切分，使得左边的最大值小于右边的最小值，对可切分的位置进行二分查找，递归进行，时间复杂度为O(log(min(m, n)))

```python
class Solution:
    def findMedianSortedArrays(self, nums1: List[int], nums2: List[int]):
        m = len(nums1)
        n = len(nums2)
        l = (m+n) // 2
        if m == 0:
            if n%2 == 0:
                return (nums2[l-1]+nums2[l]) / 2
            else:
                return nums2[l]
        if n == 0:
            if m%2 == 0:
                return (nums1[l-1]+nums1[l]) / 2
            else:
                return nums1[l]
        def helper(left, right):
            mid = (left+right) // 2
            left_max = max(nums1[mid-1], nums2[l-mid-1]) if l-mid-1 >= 0 and mid > 0 else (nums1[mid-1] if mid > 0 else nums2[l-mid-1])
            right_min = min(nums1[mid], nums2[l-mid]) if l-mid <= n-1 and mid < m else (nums1[mid] if mid < m else nums2[l-mid])
            if left_max <= right_min:
                if (m+n)%2 == 0:
                    return (left_max+right_min)/2
                else:
                    return right_min
            else:
                bottom = nums2[l-mid-1] if l-mid-1 >= 0 else -inf
                top = nums1[mid] if mid <= m-1 else inf
                if bottom > top:
                    return helper(mid+1, right)
                else:
                    return helper(left, mid - 1)
        return helper(max(l-n, 0), min(m, l))
        
        '''   
        从max(l-n,0)开始切，切到min(m,l)+1，二分查找
        m > l 时，从l-n开始切，可以切到l后面
        m < l 时，从0开始切，切到m后面
        m = l 时，从0开始切，切到l后面
        m+n偶数时 返回左边最大的和右边最小的平均值
        m+n奇数时 返回右边最小的
        '''
```

# 29 两数相除
> 给定两个整数，被除数 dividend 和除数 divisor。将两数相除，要求不使用乘法、除法和 mod 运算符。
返回被除数 dividend 除以除数 divisor 得到的商。
整数除法的结果应当截去（truncate）其小数部分，例如：truncate(8.345) = 8 以及 truncate(-2.7335) = -2

**解法：** 首先判断越界的情况，处理商的符号，并将除数和被除数转化为整数处理。对除数不断倍加，直到大于被除数的一半，记录结果，并将被除数减去除数的倍加值，递归

```python
class Solution:
    def divide(self, dividend: int, divisor: int) -> int:
        if dividend == -2**31 and divisor == -1:
            return 2**31-1
        return self.divide_abs(abs(dividend), abs(divisor)) if (dividend > 0 and divisor > 0) or (dividend < 0 and divisor < 0) else 0-self.divide_abs(abs(dividend), abs(divisor))
    def divide_abs(self, dividend: int, divisor: int) -> int:
        if dividend < divisor:
            return 0
        if dividend == divisor:
            return 1
        divisor_temp = divisor
        result_temp = 1
        while dividend > divisor_temp + divisor_temp:
            divisor_temp += divisor_temp
            result_temp += result_temp
        return result_temp + self.divide_abs(dividend - divisor_temp, divisor)
```

# 33 搜索旋转排序数组
> 假设按照升序排序的数组在预先未知的某个点上进行了旋转。( 例如，数组 [0,1,2,4,5,6,7] 可能变为 [4,5,6,7,0,1,2] )。搜索一个给定的目标值，如果数组中存在这个目标值，则返回它的索引，否则返回 -1 。
你可以假设数组中不存在重复的元素，算法时间复杂度必须是 O(log n) 级别。

**解法：** 二分查找，注意相等情况的判断

```
目标值在数组后半段
    中间值大于等于开头值，去后半段找
    中间值小于开头值，和target比
        小于target，去后半段找
        大于等于target，去前半段找
目标值在数组前半段
    中间值大于等于开头值，和target比
        小于target，去后半段找
        大于等于target，去前半段找
    中间值小于开头值，去前半段找
```

```python
class Solution:
    def search(self, nums: List[int], target: int) -> int:
        if len(nums) == 0:
            return -1
        return self.binary_search(nums, target, 0, len(nums)-1)
    def binary_search(self, nums: List[int], target:int, left:int, right:int):
        if nums[left] == target:
            return left
        if left == right:
            return -1
        mid = (left+right)//2
        if nums[left] > target:
            if nums[mid] >= target and nums[mid] < nums[left]:
                return self.binary_search(nums, target, left, mid)
            else:
                return self.binary_search(nums, target, mid+1, right)
        if nums[left] < target:
            if nums[mid] < target and nums[mid] >= nums[left]:
                return self.binary_search(nums, target, mid+1, right)
            else:
                return self.binary_search(nums, target, left, mid)
```

# 34 在排序数组中查找元素的第一个和最后一个位置
> 给定一个按照升序排列的整数数组 nums，和一个目标值 target。找出给定目标值在数组中的开始位置和结束位置的索引值。
你的算法时间复杂度必须是 O(log n) 级别。
如果数组中不存在目标值，返回 [-1, -1]。

**解法：** 递归分别查找开始位置和结束位置，查找结束位置时从开始位置开始查找，而非从0位置

```python
class Solution:
    def searchRange(self, nums: List[int], target: int) -> List[int]:
        if len(nums) == 0 or nums[0] > target or nums[len(nums)-1] < target:
            return [-1, -1]
        left = self.binarySearchFirst(nums, target, 0, len(nums)-1)
        if left == -1:
            return [-1, -1]
        right = self.binarySearchLast(nums, target, left, len(nums)-1)
        return [left, right]
    def binarySearchFirst(self, nums: List[int], target: int, left: int, right: int) -> List[int]:
        if nums[left] == target:
            return left
        if left == right:
            return -1
        mid = (left+right)//2
        if nums[mid] >= target:
            return self.binarySearchFirst(nums, target, left, mid)
        else:
            return self.binarySearchFirst(nums, target, mid+1, right)
    def binarySearchLast(self, nums: List[int], target: int, left: int, right: int) -> List[int]:
        if nums[right] == target:
            return right
        if left == right:
            return -1
        mid = (left+right)//2 + 1 if (left+right)%2 == 1 else (left+right)//2
        if nums[mid] > target:
            return self.binarySearchLast(nums, target, left, mid-1)
        else:
            return self.binarySearchLast(nums, target, mid, right)
```

# 35 搜索插入位置
> 给定一个排序数组和一个目标值，在数组中找到目标值，并返回其索引。如果目标值不存在于数组中，返回它将会被按顺序插入的位置。
你可以假设数组中无重复元素。

**解法：** 二分查找进行搜索

```python
class Solution:
    def searchInsert(self, nums: List[int], target: int) -> int:
        if target <= nums[0]:
            return 0
        if target > nums[len(nums)-1]:
            return len(nums)
        left = 0
        right = len(nums) - 1
        while right > left + 1:
            mid = (left+right)//2
            if nums[mid] < target:
                left = mid
            else:
                right = mid
        if nums[left] == target:
            return left
        else:
            return right
```

# 50 Pow(x, n)
> 实现 pow(x, n) ，即计算 x 的 n 次幂函数。
说明:
-100.0 < x < 100.0
n 是 32 位有符号整数，其数值范围是 [−231, 231 − 1] 。

**解法：** 递归进行，时间复杂度为 log(n)，空间复杂度为 log(n)
**注意：** 采用递归方法比用循环好

```python
class Solution:
    def myPow(self, x: float, n: int) -> float:
        if n < 0:
            return 1/self.PowHelper(x, -n)
        return self.PowHelper(x, n)
    def PowHelper(self, x:float, n:int):
        if n == 0:
            return 1
        y = self.PowHelper(x, n//2)
        return y*y if n%2 == 0 else y*y*x
```

# 69 x 的平方根
> 实现 int sqrt(int x) 函数。计算并返回 x 的平方根，其中 x 是非负整数。
由于返回类型是整数，结果只保留整数的部分，小数部分将被舍去。

**解法：** 二分查找，时间复杂度 log(n)，空间复杂度 1

```python
class Solution:
    def mySqrt(self, x: int) -> int:
        left = 0
        right = x
        while right**2 > x:
            mid = math.ceil((left+right)/2)
            if mid**2 > x:
                right = mid - 1
            else:
                left = mid
        return right
```

