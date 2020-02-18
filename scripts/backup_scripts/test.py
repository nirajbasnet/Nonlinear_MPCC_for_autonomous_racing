#
# import numpy as np
# from scipy.interpolate import splprep, splev
# import matplotlib.pyplot as plt
# import time
# # define pts from the question
#
#
# def find_vx_vy(v,x1,y1,x2,y2):
#     theta = np.arctan((y2-y1)/(x2-x1))
#     vx = v* np.cos(theta)
#     vy = v* np.sin(theta)
#     return vx,vy,x2-x1,y2-y1
#
# print(find_vx_vy(2,23.79,19.09,38.58,8.732))
# print(find_vx_vy(3,20.06,16.6066,-11.70,-17.8292))
# print(find_vx_vy(2,4.137,0.063,15.227,11.564))
#
# x= np.array([2,7,3,5,98,41,20,4,10])
# select_indices = np.where(x%2==0)[0]
#
# print select_indices

'''
Suppose an array sorted in ascending order is rotated at some pivot unknown to you beforehand.

(i.e., [0,1,2,4,5,6,7] might become [4,5,6,7,0,1,2]).

You are given a target value to search. If found in the array return its index, otherwise return -1.

You may assume no duplicate exists in the array.

Your algorithm's runtime complexity must be in the order of O(log n).

'''

import time


class Solution:

    #correct way if initializing
    left = 0
    right = None

    # def search(self, nums, target):
    #
    #     """
    #     :type nums: List[int]
    #     :type target: int
    #     :rtype: int
    #     """
    #     if self.right == None:
    #         self.right = len(nums) - 1
    #
    #     import math
    #     while self.right >= self.left: #needed to return -1 when item not found
    #
    #         #Do this to avoid maximum recursion depth since self.left and self.right intersect in this code
    #         if self.right-self.left <= 2:
    #
    #             for i in range(self.right-self.left+1):
    #
    #                 if nums[i + self.left] == target:
    #                     return i + self.left
    #
    #             return -1
    #
    #         #correct way of getting mid
    #         mid = int(self.right - math.ceil((self.right - self.left) / 2))
    #
    #
    #         if nums[mid] == target:
    #             return mid
    #
    #         if nums[mid] > target:
    #
    #             if nums[self.left] > target:
    #                 self.left = mid
    #                 return self.search(nums, target)
    #             else:
    #                 self.right = mid
    #                 return self.search(nums, target)
    #         else:
    #             if nums[self.right] > target:
    #                 self.left = mid
    #                 return self.search(nums, target)
    #             else:
    #                 self.right = mid
    #                 return self.search(nums, target)
    #     return -1

    def search(self, nums, target):
        self.left = 0
        self.right = len(nums)
        mid =0
        if nums==[]:
            return -1
        elif len(nums)==1:
            if nums[mid]==target:
                return mid
            else :
                return -1
        elif self.left-self.right==1:
            for i in range(self.left,self.right+1):
                if nums[i]==target:
                    return i
            return -1
        mid = (self.left+self.right)/2




if __name__ == "__main__":
    sol = Solution()
    print(sol.search([4,5,6,7,0,1,2],0))
    #print(sol.search([1],1))
    #print(sol.search([1,3,5],0))
