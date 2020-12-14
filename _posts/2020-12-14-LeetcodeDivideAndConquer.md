---
title: 力扣刷题 | 分治问题解法、例题及代码
author: 钟欣然
date: 2020-12-12 00:52:00 +0800
categories: [力扣刷题, 按问题分类]
math: true
mermaid: true
---

# 总述
在计算机科学中，分治法是构建基于多项分支递归的一种很重要的算法范式。字面上的解释是「分而治之」，就是把一个复杂的问题分成两个或更多的相同或相似的子问题，直到最后子问题可以简单的直接求解，原问题的解即子问题的解的合并。

这个技巧是很多高效算法的基础，如排序算法（快速排序、归并排序）、傅立叶变换（快速傅立叶变换）。

![在这里插入图片描述](https://img-blog.csdnimg.cn/20200925150027700.png)

另一方面，理解及设计分治法算法的能力需要一定时间去掌握。正如以归纳法去证明一个理论，为了使递归能够推行，很多时候需要用一个较为概括或复杂的问题去取代原有问题。而且并没有一个系统性的方法去适当地概括问题。

分治法这个名称有时亦会用于将问题简化为只有一个细问题的算法，例如用于在已排序的列中查找其中一项的折半搜索算法。这些算法比一般的分治算法更能有效地运行。其中，假如算法使用尾部递归的话，便能转换成简单的循环。但在这广义之下，所有使用递归或循环的算法均被视作“分治算法”。因此，有些作者考虑“分治法”这个名称应只用于每个有最少两个子问题的算法。而只有一个子问题的曾被建议使用减治法这个名称。

分治算法通常以数学归纳法来验证。而它的计算成本则多数以解递归关系式来判定。

# 4 寻找两个正序数组的中位数
> 给定两个大小为 m 和 n 的正序（从小到大）数组 nums1 和 nums2。请你找出这两个正序数组的中位数，并且要求算法的时间复杂度为 O(log(m + n))。你可以假设 nums1 和 nums2 不会同时为空。

> 示例 1:
- nums1 = [1, 3]
- nums2 = [2]
- 则中位数是 2.0

>示例 2:
- nums1 = [1, 2]
- nums2 = [3, 4]
- 则中位数是 (2 + 3)/2 = 2.5

**解法1：** 分治算法，从两个数组的开头开始找，直到找到中位数位置的元素，时间复杂度$O(m+n)$，空间复杂度$O(1)$

```python
class Solution:
    def findMedianSortedArrays(self, nums1: List[int], nums2: List[int]):
        m, n = len(nums1), len(nums2)
        i, j = 0, 0
        while i + j < (m+n)//2:
            if i < m and (j == n or nums1[i] < nums2[j]):
                last_num = nums1[i]
                i += 1
            else:
                last_num = nums2[j]
                j += 1
        
        if i == m:
            next_num = nums2[j]
        elif j == n:
            next_num = nums1[i]
        else:
            next_num = min(nums1[i], nums2[j])
        
        return next_num if (m+n)%2 == 1 else (next_num+last_num)/2
```

**解法2：** 设计到$\log$，用二分查找，定义一个在$A,B$两个数组中查找第$K$个元素的函数，如果$A[k/2]<B[k/2]$，说明我们要找的答案不在$A[0:k/2]$中，以此类推，不断二分，直到找到第$K$个元素，时间复杂度为$O(\log(m+n))$，空间复杂度为$O(\log(m+n))$

```python
class Solution:
    def findMedianSortedArrays(self, nums1: List[int], nums2: List[int]):
        def getKthElement(nums1, nums2, i, j, k):
            if i >= len(nums1)-1:
                return nums2[j+k]
            if j >= len(nums2)-1:
                return nums1[i+k]
            if k == 1:
                return min(nums1[i+1], nums2[j+1])

            temp = min([len(nums1)-i-1, len(nums2)-j-1, k//2])

            if nums1[i+temp] < nums2[j+temp]:
                return getKthElement(nums1, nums2, i+temp, j, k-temp)
            else:
                return getKthElement(nums1, nums2, i, j+temp, k-temp)


        m, n = len(nums1), len(nums2)
        if (m+n)%2 == 0:
            return (getKthElement(nums1, nums2, -1, -1, (m+n)//2) + getKthElement(nums1, nums2, -1, -1, (m+n)//2+1))/2
        else:
            return getKthElement(nums1, nums2, -1, -1, (m+n)//2+1)
```

**解法3：**  对两个数组进行切分，使得左边的最大值小于右边的最小值，对可切分的位置进行二分查找，递归进行，时间复杂度为$O(\log(\min(m, n)))$，空间复杂度为$O(1)$

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

# 23 合并K个升序链表
> 给你一个链表数组，每个链表都已经按升序排列。请你将所有链表合并到一个升序链表中，返回合并后的链表。

> 示例 1：
- 输入：lists = [[1,4,5],[1,3,4],[2,6]]
- 输出：[1,1,2,3,4,4,5,6]
> - 解释：链表数组如下：
> [
>   1->4->5,
>   1->3->4,
>  2->6
> ]
> 将它们合并到一个有序链表中得到。
1->1->2->3->4->4->5->6

> 示例 2：
- 输入：lists = []
- 输出：[]

> 示例 3：
- 输入：lists = [[]]
- 输出：[]

> 提示：
- k == lists.length
- 0 <= k <= 10^4
- 0 <= lists[i].length <= 500
- -10^4 <= lists[i][j] <= 10^4
- lists[i] 按 升序 排列
- lists[i].length 的总和不超过 10^4

**解法：** 分治算法，首先定义合并两个链表的方法，然后对数组中的链表两两合并。假设有$k$个链表，每个链表的最长长度为$n$，则时间复杂度为$O(kn\log(k))$

```python
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution:
    def mergeKLists(self, lists: List[ListNode]) -> ListNode: 
        def merge2Lists(list1, list2):
            if not list1 and not list2:
                return None
            i, j = list1, list2
            ans = ListNode(-1)
            ans_head = ans
            while i and j:
                if i.val < j.val:
                    ans.next = i
                    i, ans = i.next, ans.next
                else:
                    ans.next = j
                    j, ans = j.next, ans.next
            ans.next = i if i else j

            return ans_head.next

        if len(lists) == 0:
            return None
        
        while len(lists) > 1:
            for i in range(math.ceil(len(lists)/2)):
                lists[i] = merge2Lists(lists[2*i], lists[2*i+1] if 2*i+1 < len(lists) else [])
            lists = lists[:(i+1)]
        return lists[0]
```

# 53 最大子序和
> 给定一个整数数组 nums ，找到一个具有最大和的连续子数组（子数组最少包含一个元素），返回其最大和。

> 示例:
- 输入: [-2,1,-3,4,-1,2,1,-5,4]
- 输出: 6
- 解释: 连续子数组 [4,-1,2,1] 的和最大，为 6。

**解法1：** 分治法，时间复杂度$O(n)$，空间复杂度$O(\log n)$，将每个数组分为左右两个部分，分别维护两个子数组的
- leftSum：以左端点为起点的最大子序和
- rightSum：以右端点为终点的最大子序和
- maxSum：这个子数组的最大子序和
- allSum：这个子数组的总和

将两个子数组合并起来时：
- leftSum：max（左子数组的leftSum，左子数组的allSum+右子数组的leftSum）
- rightSum：max（右子数组的rightSum，右子数组的allSum+左子数组的rightSum）
- maxSum：max（左子数组的maxSum，右子数组的maxSum，左子数组的rightSum+右子数组的leftSum）
- allSum：左子数组的allSum+右子数组的allSum

```python
class Solution:
    def maxSubArray(self, nums: List[int]) -> int:
        def get(nums, left, right):
            if right-left == 1:
                return nums[left], nums[left], nums[left], nums[left]

            mid = (left+right)//2
            leftSum1, rightSum1, maxSum1, allSum1 = get(nums, left, mid)
            leftSum2, rightSum2, maxSum2, allSum2 = get(nums, mid, right)
            return max(leftSum1, allSum1 + leftSum2), max(rightSum2, allSum2 + rightSum1), max(maxSum1, maxSum2, rightSum1+leftSum2), allSum1 + allSum2
        
        _, _, ans, _ = get(nums, 0, len(nums))
        return ans
```

**解法2：** 动态规划，时间复杂度$O(n)$，空间复杂度$O(1)$，对每个数计算current_sum和sum_max，例如：

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
    def maxSubArray(self, nums: List[int]) -> int:
        current_max,sum_max = nums[0],nums[0]
        for num in nums[1:]:
            current_max = (current_max + num) if current_max > 0 else num
            sum_max = current_max if current_max > sum_max else sum_max
        return sum_max
```

# 169 多数元素
> 给定一个大小为 n 的数组，找到其中的多数元素。多数元素是指在数组中出现次数大于 ⌊ n/2 ⌋ 的元素。你可以假设数组是非空的，并且给定的数组总是存在多数元素。

> 示例 1:
- 输入: [3,2,3]
- 输出: 3

> 示例 2:
- 输入: [2,2,1,1,1,2,2]
- 输出: 2

**解法1：** 哈希表，通过哈希表记录各元素出现的次数，时间复杂度$O(n)$，空间复杂度$O(n)$，注意在遍历数组同时使用打擂台的方法，维护最大的值，省去最后对哈希映射的遍历

```python
class Solution:
    def majorityElement(self, nums: List[int]) -> int:
        ans = {}
        ans_num = nums[0]
        for num in nums:
            ans[num] = (ans[num] + 1) if num in ans.keys() else 1
            if ans[num] > ans[ans_num]:
                ans_num = num
        return ans_num
```
**解法2：** 分治法，将数组分为左右两部分，数组的众数一定是左子数组的众数或右子数组的众数，具体是哪一个通过比较二者出现次数决定，时间复杂度$O(n\log n)$，空间复杂度$O(\log n)$

```python
class Solution:
    def majorityElement(self, nums: List[int]) -> int:
        def localMajorityElement(nums, left, right):
            if right-left == 1:
                return nums[left], 1
            
            mid = (left+right)//2
            leftMajority, leftNum = localMajorityElement(nums, left, mid)
            rightMajority, rightNum = localMajorityElement(nums, mid, right)

            leftNumInRight, rightNumInLeft = 0, 0
            for ele in nums[mid:right]:
                if ele == leftMajority:
                    leftNumInRight += 1 
            for ele in nums[left:mid]:
                if ele == rightMajority:
                    rightNumInLeft += 1
            
            if leftNum + leftNumInRight > rightNum + rightNumInLeft:
                return leftMajority, leftNum + leftNumInRight
            else:
                return rightMajority, rightNum + rightNumInLeft
            
        ans, _ = localMajorityElement(nums, 0, len(nums))
        return ans
```

# 215 数组中的第K个最大元素
> 在未排序的数组中找到第 k 个最大的元素。请注意，你需要找的是数组排序后的第 k 个最大的元素，而不是第 k 个不同的元素。你可以假设 k 总是有效的，且 1 ≤ k ≤ 数组的长度。

> 示例 1:
- 输入: [3,2,1,5,6,4] 和 k = 2
- 输出: 5

>示例 2:
- 输入: [3,2,3,1,2,4,5,5,6] 和 k = 4
- 输出: 4

**解法1：** 快速排序，获取对应元素，时间复杂度$O(n\log n)$，空间复杂度$O(n\log n)$

```python
class Solution:
    def findKthLargest(self, nums: List[int], k: int) -> int:
        def quickSort(nums):
            if len(nums) == 1:
                return [nums[0]]
            
            mid = len(nums)//2
            left = quickSort(nums[:mid])
            right = quickSort(nums[mid:])

            i, j, ans = 0, 0, []
            while i < len(left) and j < len(right):
                if left[i] < right[j]:
                    ans.append(right[j])
                    j += 1
                else:
                    ans.append(left[i])
                    i += 1
            ans.extend(left[i:]) if i < len(left) else ans.extend(right[j:])

            return ans
        return quickSort(nums)[k-1]
```
**解法2：** 利用堆排序，k次将大顶堆的堆顶调至末尾（调至末尾而非直接删除，是为了降低后续调至大顶堆的复杂度），时间复杂度$O(n\log n)$，空间复杂度$O(1)$

```python
class Solution:
    def findKthLargest(self, nums: List[int], k: int) -> int:
        def heapAdjust(nums, start, end):
            temp, son = nums[start], 2*start+1
            while son <= end:
                if son < end and nums[son] < nums[son+1]:
                    son += 1
                if nums[son] <= temp:
                    break
                nums[start] = nums[son]
                start, son = son, 2*son+1
            nums[start] = temp

        for i in range(len(nums)//2-1, -1, -1):
            heapAdjust(nums, i, len(nums)-1)

        d = 1
        nums[len(nums)-1], nums[0] = nums[0], nums[len(nums)-1]
        while d < k:
            heapAdjust(nums, 0, len(nums)-d-1)
            nums[len(nums)-d-1], nums[0] = nums[0], nums[len(nums)-d-1]
            d += 1
        return nums[len(nums)-k]
```

# 218 天际线问题
> 城市的天际线是从远处观看该城市中所有建筑物形成的轮廓的外部轮廓。现在，假设您获得了城市风光照片（图A）上显示的所有建筑物的位置和高度，请编写一个程序以输出由这些建筑物形成的天际线（图B）。

![在这里插入图片描述](https://img-blog.csdnimg.cn/20200927132834779.png)

![在这里插入图片描述](https://img-blog.csdnimg.cn/20200927133005694.png)

> 每个建筑物的几何信息用三元组 [Li，Ri，Hi] 表示，其中 Li 和 Ri 分别是第 i 座建筑物左右边缘的 x 坐标，Hi 是其高度。可以保证 0 ≤ Li, Ri ≤ INT_MAX, 0 < Hi ≤ INT_MAX 和 Ri - Li > 0。您可以假设所有建筑物都是在绝对平坦且高度为 0 的表面上的完美矩形。
- 例如，图A中所有建筑物的尺寸记录为：[ [2 9 10], [3 7 15], [5 12 12], [15 20 10], [19 24 8] ] 。
输出是以 [ [x1,y1], [x2, y2], [x3, y3], ... ] 格式的“关键点”（图B中的红点）的列表，它们唯一地定义了天际线。关键点是水平线段的左端点。请注意，最右侧建筑物的最后一个关键点仅用于标记天际线的终点，并始终为零高度。此外，任何两个相邻建筑物之间的地面都应被视为天际线轮廓的一部分。
- 例如，图B中的天际线应该表示为：[ [2 10], [3 15], [7 12], [12 0], [15 10], [20 8], [24, 0] ]。

> 说明:
- 任何输入列表中的建筑物数量保证在 [0, 10000] 范围内。
- 输入列表已经按左 x 坐标 Li  进行升序排列。
- 输出列表必须按 x 位排序。
- 输出天际线中不得有连续的相同高度的水平线。例如 [...[2 3], [4 5], [7 5], [11 5], [12 7]...] 是不正确的答案；三条高度为 5 的线应该在最终输出中合并为一个：[...[2 3], [4 5], [12 7], ...]

**解法1：** 哈希表，首先记录每个位置的最大值，再生成天际线，超出时间限制

```python
class Solution:
    def getSkyline(self, buildings: List[List[int]]) -> List[List[int]]:
        if len(buildings) == 0:
            return [] 

        buildings.sort(key = lambda x: x[2], reverse = True)
        xyMap = {}
        for building in buildings:
            for i in range(building[0], building[1]+1):
                if i not in xyMap.keys():
                    xyMap[i] = building[2]
        
        leftBorder, rightBorder = min(xyMap), max(xyMap)
        ans = [[leftBorder, xyMap[leftBorder]]]
        for x in range(leftBorder+1, rightBorder+2):
            tempHeight = xyMap[x] if x in xyMap.keys() else 0
            if tempHeight > ans[-1][1]:
                ans.append([x, tempHeight])
            elif tempHeight < ans[-1][1]:
                ans.append([x-1, tempHeight])
        return ans
```
**解法2：** 分治法，若仅有一个建筑物，天际线为其左上角和右下角；有多个建筑物时，先将其分为两组，再将两组的天际线合并起来。合并方法为，检查两组的共同位置上每个关键点，维护一个当前高度currY，如果 max(左天际线高度，右天际线高度) 不等于currY，则将其添加至天际线中，并修改currY

```python
class Solution:
    def getSkyline(self, buildings: List[List[int]]) -> List[List[int]]:
        def mergeSkyline(left, right):
            if left[-1][0] < right[0][0]:
                return left + right
                
            i, j = 0, 0
            while i < len(left)-1 and right[0][0] > left[i+1][0]:
                i += 1
            
            ans = left[:i] if right[0][0] == left[i][0] and left[i][1] < right[0][1] else left[:(i+1)]
            currY = left[i][1]
            # 此步为筛选出作为关键点的x进行后续处理，若对range(right[0][0], min(right[-1][0], left[-1][0])+1)的所有x进行处理，会超出时间限制
            xList = []
            for p in range(i, len(left)):
                if left[p][0] > min(right[-1][0], left[-1][0]):
                    break
                elif left[p][0] > right[0][0]:
                    xList.append(left[p][0])
            for q in range(len(right)):
                if right[q][0] > min(right[-1][0], left[-1][0]):
                    break
                else:
                    xList.append(right[q][0])
            xList.sort()


            for x in xList:
                while i < len(left)-1 and x >= left[i+1][0]:
                    i += 1
                while j < len(right)-1 and x >= right[j+1][0]:
                    j += 1
                if max(left[i][1], right[j][1]) != currY:
                    currY = max(left[i][1], right[j][1])
                    ans.append([x, currY])

            ans.extend(left[(i+1):]) if i < len(left)-1 else ans.extend(right[(j+1):])
            return ans

        if len(buildings) == 0:
            return []
        elif len(buildings) == 1:
            return [[buildings[0][0], buildings[0][2]], [buildings[0][1], 0]]
        else:
            leftSkyline = self.getSkyline(buildings[:len(buildings)//2])
            rightSkyline = self.getSkyline(buildings[len(buildings)//2:])
            return mergeSkyline(leftSkyline, rightSkyline)

```
# 240 搜索二维矩阵
> 编写一个高效的算法来搜索 m x n 矩阵 matrix 中的一个目标值 target。该矩阵具有以下特性：
每行的元素从左到右升序排列。
每列的元素从上到下升序排列。

> 示例:
- 现有矩阵 matrix 如下：
> [
>   [1,   4,  7, 11, 15],
>   [2,   5,  8, 12, 19],
>   [3,   6,  9, 16, 22],
>   [10, 13, 14, 17, 24],
>   [18, 21, 23, 26, 30]
> ]
- 给定 target = 5，返回 true。
- 给定 target = 20，返回 false。

**解法1：** 对每行元素二分查找，时间复杂度$O(m\log n)$，空间复杂度$O(1)$

```python
class Solution:
    def searchMatrix(self, matrix, target):
        """
        :type matrix: List[List[int]]
        :type target: int
        :rtype: bool
        """
        def binarySearch(lst, target):
            if len(lst) == 0:
                return False
            
            left, right = 0, len(lst)
            while right-left >= 1:
                mid = (right+left)//2
                if lst[mid] == target:
                    return True
                elif lst[mid] > target:
                    right = mid
                else:
                    left = mid+1
            return False

        if len(matrix) == 0 or len(matrix[0]) == 0:
            return False

        for i in range(len(matrix)):
            if matrix[i][0] > target:
                return False
            if matrix[i][len(matrix[i])-1] >= target and binarySearch(matrix[i], target):
                return True
            
        return False
```

**解法2：** 迭代对角线元素，对对角线元素所在行列（不包括对角线以后的数据）进行二分查找，时间复杂度$O(\log (\max (m,n)!))$，空间复杂度$O(1)$

```python
class Solution:
    def searchMatrix(self, matrix, target):
        """
        :type matrix: List[List[int]]
        :type target: int
        :rtype: bool
        """
        def binarySearch(lst, target):
            if len(lst) == 0:
                return False
            
            left, right = 0, len(lst)
            while right-left >= 1:
                mid = (right+left)//2
                if lst[mid] == target:
                    return True
                elif lst[mid] > target:
                    right = mid
                else:
                    left = mid+1
            return False

        m = len(matrix)
        n = len(matrix[0]) if m > 0 else 1
        if m == 0 or n == 0:
            return False
        i = 0
        while i < max(m, n):
            if matrix[min(i, m-1)][min(i, n-1)] == target:
                return True
            elif matrix[min(i, m-1)][min(i, n-1)] > target:
                if (i < m and binarySearch(matrix[i][:i], target)) or (i < n and binarySearch([matrix[x][i] for x in range(min(i, m-1)+1)], target)):
                    return True
            i += 1
        
        return False
```

**解法3：** 对对角线进行迭代，找到$matrix[i-1][i-1]<target<matrix[i][i]$的位置，根据此位置将大矩阵分为四部分，则目标元素仅可能出现在左下和右上矩阵中，递归求解，时间复杂度$O(n\log n)$，空间复杂度$O(\log n)$，复杂度计算方法参见[力扣](https://leetcode-cn.com/problems/search-a-2d-matrix-ii/solution/sou-suo-er-wei-ju-zhen-ii-by-leetcode-2/)

```python
class Solution:
    def searchMatrix(self, matrix, target):
        """
        :type matrix: List[List[int]]
        :type target: int
        :rtype: bool
        """
        m = len(matrix)
        n = len(matrix[0]) if m > 0 else 1
        if m == 0 or n == 0:
            return False

        if matrix[0][0] == target:
            return True
        elif matrix[0][0] > target:
            return False
        
        i = 0
        for i in range(1, min(m, n)):
            if matrix[i][i] == target:
                return True
            elif matrix[i][i] > target:
                break
        if i == min(m, n)-1 and matrix[i][i] < target:
            i += 1
        return self.searchMatrix([matrix[x][i:] for x in range(i)], target) or self.searchMatrix([matrix[x][:i] for x in range(i, m)], target)
```
**解法4：** 双指针法，从左下角开始，若当前值大于目标值，行指针-1，若小于目标值，列指针+1，直到找到目标值或到达边界。时间复杂度$O(m+n)$，空间复杂度$O(1)$

```python
class Solution:
    def searchMatrix(self, matrix, target):
        """
        :type matrix: List[List[int]]
        :type target: int
        :rtype: bool
        """
        m = len(matrix)
        n = len(matrix[0]) if m > 0 else 1
        if m == 0 or n == 0:
            return False

        i, j = m-1, 0
        while i > -1 and j < n:
            if matrix[i][j] > target:
                i -= 1
            elif matrix[i][j] < target:
                j += 1
            else:
                return True
        
        return False
```

