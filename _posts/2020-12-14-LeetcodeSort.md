---
title: 力扣刷题 | 排序问题解法、例题及代码（包含十大排序算法的描述、复杂度和 Python 实现）
author: 钟欣然
date: 2020-12-12 00:51:00 +0800
categories: [力扣刷题, 按问题分类]
math: true
mermaid: true
---

# 总述
## 十大排序算法及分类

![在这里插入图片描述](https://img-blog.csdnimg.cn/20200923192519679.png)

## 算法描述
- **直接选择排序**：为每一个位置选择当前最小的元素。首先从第1个位置开始对全部元素进行选择，选出全部元素中最小的放在该位置，再对第2个位置进行选择，在剩余元素中选择最小的放在该位置，以此类推，重复进行“最小元素”的选择，直至完成第$n-1$个位置的元素选择，则第n个位置就只剩唯一的最大元素
- **堆排序**：利用堆这种数据结构所设计的一种排序算法。堆是一个近似完全二叉树的结构，并同时满足堆积的性质，子结点的键值或索引总是小于（或者大于）它的父节点。首先将整个数组调整为大顶堆，调换第一个元素（最大的元素）和最后一个元素的位置，然后将从头到倒数第二个元素组成的数组调整为大顶堆，调换第一个元素（最大的元素）金额倒数第二个元素的位置，以此类推，直到堆只有一个元素
- **直接插入排序**：从第二个元素开始，如果大于第一个元素，则将其插入第一个位置，然后观察第三个元素，依次检查其是否大于第二个元素和第一个元素，将其插入合适的位置，以此类推，直到插入完最后一个元素
- **希尔排序**：又称为缩小增量排序，把记录按下标的一定增量分组，对每组使用直接插入排序算法排序；随着增量逐渐减少，每组包含的元素越来越多，当增量减至 1 时，所有元素恰被分成一组
- **冒泡排序**：把较小的元素往前调或者把较大的元素往后调。以较大元素向后调为例，从头至尾，依次检查当前元素和下一元素的大小，如果当前元素大于下一元素，则交换二者的位置，这种检查要进行$n$轮，第$i$轮依次检查从第一个到第$n-i$个元素
- **快速排序**：设定一个基准值，通过一次循环将数组的所有元素分为两组，一组比基准值大，一组比基准值小，对两组递归调用函数完成组内的排序
- **归并排序**：把序列划分为2个短序列，对短序列递归调用函数完成组内排序，再将有序的短序列进行排序为长序列
- **计数排序**：空间换时间，创建一个长度为 数组最大值-数组最小值+1 的0数组，遍历数组，将0数组中索引为 对应元素-数组最小值 的位置加一，再遍历新数组，输出元素大于0的位置的索引值，即得到排序后的数组
- **桶排序**：先将所有元素分入有序的桶，桶内也可以通过递归再次分桶或调用其他排序方法（如快速排序）进行排序
- **基数排序**：依次根据个位数字、十位数字、百位数字等进行分桶。第一次按照个位数字排序后，将各桶合并，然后再按十位数字排序，此时相同十位数字的桶中个位小的在前面，个位大的在后面，以此类推，即得到排序后的数组。

## 复杂度

![在这里插入图片描述](https://img-blog.csdnimg.cn/20200923192551907.png)

# Python 实现
## 选择排序
```python
def SelectSort(lst):
    if len(lst) <= 1:
        return lst
    
    for i in range(len(lst)):
        temp = i
        for j in range(i+1, len(lst)):
            if lst[j] < lst[temp]:
                temp = j
        lst[i], lst[temp] = lst[temp], lst[i]
            
    return lst
```
## 堆排序

```python
def HeapSort(lst):
                    
    def heapAdjust(lst, start, end):
        temp = lst[start]
        son = 2 * start + 1
        while son <= end:
            if son < end and lst[son] < lst[son+1]:
                son += 1
            if lst[son] <= temp:
                break
            lst[start] = lst[son]
            start, son = son, 2*son+1
        lst[start] = temp
            
    
    if len(lst) <= 1:
        return lst
    
    for i in range(len(lst)//2-1, -1, -1):
        heapAdjust(lst, i, len(lst)-1)
    for i in range(len(lst)-1, 0, -1):
        lst[i], lst[0] = lst[0], lst[i]
        heapAdjust(lst, 0, i-1)

    return lst
```
## 插入排序

```python
def InsertSort(lst):
    if len(lst) <= 1:
        return lst
    
    for i in range(1, len(lst)):
        j = i
        target = lst[i]
        while j > 0 and lst[j-1] > target:
            lst[j] = lst[j-1]
            j -= 1
        lst[j] = target
        
    return lst
```
## 希尔排序

```python
def ShellSort(lst):
    if len(lst) <= 1:
        return lst
    
    d = len(lst)//2
    while d >= 1:
        for i in range(d, len(lst)):
            k = d
            temp = lst[i]
            while i-k >=0 and temp < lst[i-k]:
                lst[i-k+d] = lst[i-k]
                k += d
            lst[i-k+d] = temp
        d = d//2
    return lst
```
## 冒泡排序

```python
def BubbleSort(lst):
    if len(lst) <= 1:
        return lst
    
    for i in range(len(lst)-1):
        didSwap = False
        for j in range(len(lst)-1-i):
            if lst[j] > lst[j+1]:
                lst[j], lst[j+1] = lst[j+1], lst[j]
                didSwap = True
        if not didSwap:
            return lst
    
    return lst
```
## 快速排序

```python
def QuickSort(lst):
    
    def partition(lst, left, right):
        # right > left
        key = left
        i = left + 1
        while i <= right:
            if lst[i] < lst[key]:
                j = i-1
                temp = lst[i]
                while j >= key:
                    lst[j+1] = lst[j]
                    j -= 1
                lst[key] = temp
                key += 1
            i += 1
        return key
    
    def quickSort(lst, left, right):
        if left >= right:
            return lst[left:(right+1)]
        
        key = partition(lst, left, right)
        quickSort(lst, left, key-1)
        quickSort(lst, key+1, right)
        
    quickSort(lst, 0, len(lst)-1)
    return lst
```
## 归并排序

```python
def MergeSort(lst):
    if len(lst) <= 1:
        return lst
    
    left = MergeSort(lst[:(len(lst)//2)])
    right = MergeSort(lst[(len(lst)//2):])
    
    i, j = 0, 0
    res = []
    while i < len(left) and j < len(right):
        if left[i] < right[j]:
            res.append(left[i])
            i += 1
        else:
            res.append(right[j])
            j += 1
    res.extend(left[i:]) if i < len(left) else res.extend(right[j:])
    return res
```
## 计数排序

```python
def CountSort(lst):
    if len(lst) <= 1:
        return lst
    
    lst_min = min(lst)
    lst_max = max(lst)
    
    temp = [0]*(lst_max-lst_min+1)
    for num in lst:
        temp[num-lst_min] += 1
        
    res = []
    for i in range(len(temp)):
        while temp[i] != 0:
            res.append(i)
            temp[i] -= 1
            
    return res
```
## 桶排序

```python
def BucketSort(lst):
    if len(lst) <= 1:
        return lst
    
    buckets = [[] for i in range(max(lst)//10+1)]
    for num in lst:
        buckets[num//10].append(num)
        
    res = []
    for bucket in buckets:
        res.extend(QuickSort(bucket))
    return res
```
## 基数排序

```python
import math

def RadixSort(lst):
    if len(lst) <= 1:
        return lst
    
    d = int(math.log10(max(lst)))+1
    
    for i in range(d):
        temp = [[] for i in range(10)]
        for num in lst:
            temp[(num//(10**i))%10].append(num)
        lst = []
        for arr in temp:
            for num in arr:
                lst.append(num)
    
    return lst
```

# 56 合并区间
> 给出一个区间的集合，请合并所有重叠的区间。

 > 示例 1:
- 输入: intervals = [[1,3],[2,6],[8,10],[15,18]]
- 输出: [[1,6],[8,10],[15,18]]
- 解释: 区间 [1,3] 和 [2,6] 重叠, 将它们合并为 [1,6].

 > 提示：
- intervals[i][0] <= intervals[i][1]

**解法：** 首先对原区间按照区间左端点进行排序，将结果存储在 merge 中。遍历原区间列表，依次决定和 merge 的最后一个合并还是像 merge 中添加新区间。

```python
class Solution:
    def merge(self, intervals: List[List[int]]) -> List[List[int]]:
        if len(intervals) <= 1:
            return intervals
        
        intervals.sort(key = lambda x: x[0]) # 注意此处的排序

        merge = [intervals[0]]
        for interval in intervals:
            if interval[0] <= merge[-1][1]:
                merge[-1][1] = max(merge[-1][1], interval[1])
            else:
                merge.append(interval)
        return merge
```

# 57 插入区间
> 给出一个无重叠的 ，按照区间起始端点排序的区间列表。
在列表中插入一个新的区间，你需要确保列表中的区间仍然有序且不重叠（如果有必要的话，可以合并区间）。

> 示例 1：
- 输入：intervals = [[1,3],[6,9]], newInterval = [2,5]
- 输出：[[1,5],[6,9]]

> 示例 2：
- 输入：intervals = [[1,2],[3,5],[6,7],[8,10],[12,16]], newInterval = [4,8]
- 输出：[[1,2],[3,10],[12,16]]
- 解释：这是因为新的区间 [4,8] 与 [3,5],[6,7],[8,10] 重叠。

**解法：** 贪心算法，依次检查新区间左端点在原区间中的位置（遍历每个原区间，检查新区间左端点是否小于等于原区间左端点/又端点），找到后再向后检查新区间又端点在原区间中的位置，方法同上，最后返回结果

```python
class Solution:
    def insert(self, intervals: List[List[int]], newInterval: List[int]) -> List[List[int]]:
        if len(intervals) == 0:
            return [newInterval]

        i = 0
        while i < len(intervals)-1:
            if newInterval[0] <= intervals[i][0]:
                if newInterval[1] < intervals[i][0]:
                    return intervals[:i] + [newInterval] + intervals[i:]
                else:
                    j = i
                    while j+1 < len(intervals) and newInterval[1] >= intervals[j+1][0]:
                        j += 1
                    return intervals[:i] + [[newInterval[0], max(newInterval[1], intervals[j][1])]] + intervals[(j+1):]
            elif newInterval[0] <= intervals[i][1]:
                if newInterval[1] < intervals[i][1]:
                    return intervals
                else:
                    j = i
                    while j+1 < len(intervals) and newInterval[1] >= intervals[j+1][0]:
                        j += 1
                    return intervals[:i] + [[intervals[i][0], max(newInterval[1], intervals[j][1])]] + intervals[(j+1):]
            i += 1

        # 最后一个
        if newInterval[1] < intervals[len(intervals)-1][0]:
            return intervals[:(len(intervals)-1)] + [newInterval] + [intervals[len(intervals)-1]]
        elif newInterval[0] > intervals[len(intervals)-1][1]:
            return intervals + [newInterval]
        else:
            return intervals[:(len(intervals)-1)] + [[min(newInterval[0], intervals[len(intervals)-1][0]), max(newInterval[1], intervals[len(intervals)-1][1])]]
```

# 75 颜色分类
> 给定一个包含红色、白色和蓝色，一共 n 个元素的数组，原地对它们进行排序，使得相同颜色的元素相邻，并按照红色、白色、蓝色顺序排列。此题中，我们使用整数 0、 1 和 2 分别表示红色、白色和蓝色。

> 注意:
- 不能使用代码库中的排序函数来解决这道题。

> 示例:
- 输入: [2,0,2,1,1,0]
- 输出: [0,0,1,1,2,2]

> 进阶：
一个直观的解决方案是使用计数排序的两趟扫描算法。
首先，迭代计算出0、1 和 2 元素的个数，然后按照0、1、2的排序，重写当前数组。
你能想出一个仅使用常数空间的一趟扫描算法吗？

**解法：** 利用三个指针，分别维护0的右边界、2的左边界和当前考虑的元素，逐个考虑当前元素，为0则和0的右边界交换，为2则和2的左边界交换

```python
class Solution:
    def sortColors(self, nums: List[int]) -> None:
        """
        Do not return anything, modify nums in-place instead.
        """
        if len(nums) <= 1:
            return nums

        p0, p2, curr = 0, len(nums)-1, 0
        while curr <= p2:
            if nums[curr] == 0:
                nums[p0], nums[curr] = nums[curr], nums[p0]
                p0 += 1
                curr += 1
            elif nums[curr] == 2:
                nums[p2], nums[curr] = nums[curr], nums[p2]
                p2 -= 1
            else:
                curr += 1
        
        return None
```

# 147 对链表进行插入排序

> 对链表进行插入排序。

![在这里插入图片描述](https://img-blog.csdnimg.cn/20200920115523218.gif)

插入排序的动画演示如上。从第一个元素开始，该链表可以被认为已经部分排序（用黑色表示）。每次迭代时，从输入数据中移除一个元素（用红色表示），并原地将其插入到已排好序的链表中。

> 插入排序算法：
- 插入排序是迭代的，每次只移动一个元素，直到所有元素可以形成一个有序的输出列表。
- 每次迭代中，插入排序只从输入数据中移除一个待排序的元素，找到它在序列中适当的位置，并将其插入。
- 重复直到所有输入数据插入完为止。

> 示例 1：
- 输入: 4->2->1->3
- 输出: 1->2->3->4

> 示例 2：
- 输入: -1->5->3->4->0
- 输出: -1->0->3->4->5

**解法：** 每次检查一个元素，若这个元素大于上一个元素则跳过，否则记录这个元素，将链表跳过当前元素重新连接，从头开始检查链表，找到合适位置插入当前元素。

```python
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, x):
#         self.val = x
#         self.next = None

class Solution:
    def insertionSortList(self, head: ListNode) -> ListNode:
        if not head or not head.next:
            return head

        result = ListNode(-1)
        result.next = head

        while head.next:
            if head.val <= head.next.val:
                head = head.next
            else:
                new = result
                curr_val = head.next.val
                head.next = head.next.next
                while new.next.val <= curr_val:
                    new = new.next
                new.next, new.next.next = ListNode(curr_val), new.next
        return result.next
```
# 148 排序链表
> 在 O(n log n) 时间复杂度和常数级空间复杂度下，对链表进行排序。

> 示例 1:
- 输入: 4->2->1->3
- 输出: 1->2->3->4

> 示例 2:
- 输入: -1->5->3->4->0
- 输出: -1->0->3->4->5

**解法1：** 不断二分链表进行递归，再将两个排序过的链表合并，时间复杂度 $O(n\log n)$，空间复杂度$O(n)$

![在这里插入图片描述](https://img-blog.csdnimg.cn/20200920134033384.png)

```python
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution:
    def sortList(self, head: ListNode) -> ListNode:
        if not head or not head.next:
            return head

        slow, fast = head, head.next
        while fast and fast.next:
            slow, fast = slow.next, fast.next.next
        mid, slow.next = slow.next, None

        left_rank, right_rank = self.sortList(head), self.sortList(mid)

        result = ListNode(-1)
        result_head = result
        while left_rank and right_rank:
            if left_rank.val <= right_rank.val:
                result.next, left_rank = left_rank, left_rank.next
            else:
                result.next, right_rank = right_rank, right_rank.next
            result = result.next
        # 注意这里的写法，避免在while中每次检查left_rank和right_rank是否为空
        result.next = left_rank if left_rank else right_rank
        return result_head.next
                
```

**解法2：** 要达到常数级别的空间复杂度，就不能使用递归，因此不能有解法1中的分割部分，而是直接进行合并部分，每次合并的单位依次为1、2、4、8、16，此时时间复杂度 $O(n\log n)$，空间复杂度$O(1)$

![在这里插入图片描述](https://img-blog.csdnimg.cn/20200920134010129.png)

注意：实现较为复杂
```python
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution:
    def sortList(self, head: ListNode) -> ListNode:
        if not head or not head.next:
            return head

        l, length, intv = head, 0, 1
        while l:
            length += 1
            l = l.next

        res = ListNode(-1)
        res.next = head

        while intv < length:
            temp, temp1 = res, res.next
            while temp1:
                # 获取h1
                h1, i = temp1, intv
                while temp1 and i:
                    temp1, i = temp1.next, i-1
                if i:
                    break
                
                # 获取h2
                h2 ,j = temp1, intv
                while temp1 and j:
                    temp1, j = temp1.next, j-1
                
                # 对h1、h2排序
                c1, c2 = intv, intv-j
                while c1 and c2:
                    if h1.val <= h2.val:
                        temp.next, h1, c1 = h1, h1.next, c1-1
                    else:
                        temp.next, h2, c2 = h2, h2.next, c2-1
                    temp = temp.next
                temp.next = h1 if c1 else h2
                while c1>0 or c2>0:
                    temp, c1, c2 = temp.next, c1-1, c2-1
                temp.next = temp1
            intv *= 2

        return res.next
```

# 164 最大间距
> 给定一个无序的数组，找出数组在排序之后，相邻元素之间最大的差值。如果数组元素个数小于 2，则返回 0。

> 示例 1:
- 输入: [3,6,9,1]
- 输出: 3
- 解释: 排序后的数组是 [1,3,6,9], 其中相邻元素 (3,6) 和 (6,9) 之间都存在最大差值 3。

> 示例 2:
- 输入: [10]
- 输出: 0
- 解释: 数组元素个数小于 2，因此返回 0。

> 说明:
- 你可以假设数组中所有元素都是非负整数，且数值在 32 位有符号整数范围内。
- 请尝试在线性时间复杂度和空间复杂度的条件下解决此问题。

**解法1：** 比较排序，再找最大间距，时间复杂度 $O(n\log n)$，空间复杂度$O(1)$

```python
class Solution:
    def maximumGap(self, nums: List[int]) -> int:
        if len(nums) < 2:
            return 0
        
        nums.sort()
        res = 0
        for i in range(1, len(nums)):
            res = max(res, nums[i]-nums[i-1])
        return res
```
**解法2：** 计数排序，空间换时间，创建一个长度为数组中最大值减去最小值的0数组，数组中对应的元素减去最小值的位置修改为1，遍历新数据，关注对应元素不为0的索引，找到最大间距，此时时间复杂度 $O(n+k)$，空间复杂度$O(k)$，k为数组中最大值-最小值

```python
class Solution:
    def maximumGap(self, nums: List[int]) -> int:
        if len(nums) < 2:
            return 0
        
        nums_min = min(nums)
        nums_max = max(nums)
        
        temp = [0]*(nums_max-nums_min+1)
        for num in nums:
            temp[num-nums_min] += 1
        
        pre, res = -1, 0
        for i in range(len(temp)):
            if temp[i] != 0:
                if pre != -1:
                    res = max(res, i-pre)
                pre = i
        return res
```
**解法3：** 桶+鸽笼原理，时间复杂度$O(n+b)≈O(n)$，空间复杂度$O(b)$（每个桶只需要存储最大和最小元素）

考虑数组中元素等距的情况，则间距为 $t = (max - min)/(n-1)$，我们以 $b(0 \leq b \leq t)$为窗口大小，创建多个有序的桶，桶内元素间距小于等于 t，因此我们只需要比较桶间间距即可。

> 鸽笼原理：n 个物品放入 m 个容器中，如果 n > m，那么一定有一个容器装有至少两个物品。


```python
class Solution:
    def maximumGap(self, nums: List[int]) -> int:
        if len(nums) < 2:
            return 0
        
        nums_min = min(nums)
        nums_max = max(nums)
        if nums_max == nums_min:
            return 0
        
        t = max(1, math.floor((nums_max-nums_min)/(len(nums)-1)))
        b = math.ceil((nums_max-nums_min)/t)+1

        temp= [[] for i in range(b)]
       
        for num in nums:
            k = math.floor((num-nums_min)/t)
            temp[k] = [min(min(temp[k]), num), max(max(temp[k]), num)] if temp[k] else [num, num]
        res = max(temp[0])-min(temp[0])
        for i in range(b):
            j = i+1
            while j < b and not temp[j]:
                j += 1
            if j < b and temp[i] and temp[j]:
                res = max(res, temp[j][0]-temp[i][1])
        return res
```

