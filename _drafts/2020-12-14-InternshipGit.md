---
title: 算法实习 | Git使用指南
author: 钟欣然
date: 2020-12-12 00:46:00 +0800
categories: [实习收获, 字节飞书]
math: true
mermaid: true
---



新手入门建议查看完整版[Git教程](https://www.liaoxuefeng.com/wiki/896043488029600
)，此处仅做整理

## 简介
Git是目前世界上最先进的分布式版本控制系统，能自动记录每次文件的改动，还可以多人协作编辑。

集中式版本控制系统中，版本库是集中存放在中央服务器的，用户从中央服务器取得最新版本，修改后再将文件推送给中央服务器，必须联网才能工作。著名的集中式版本控制系统有CVS、SVN。
![在这里插入图片描述](https://img-blog.csdnimg.cn/2020022818323423.png#pic_center =400x200)<center>集中式版本控制系统示意图</center><br>
分布式版本控制系统没有中央服务器，每个人的电脑上都是一个完整的版本库，只需要将修改推送给协作者即可完成协作，安全性更高（一个人的电脑损坏了是需要从其他人那里同步即可）。分布式版本控制系统通常也有一台充当中央服务器的电脑，方便交换大家的修改。
![在这里插入图片描述](https://img-blog.csdnimg.cn/20200228185624272.png#pic_center =300x200)
<center>分布式版本控制系统示意图</center><br>

## 安装Git
在Windows上安装:

- 从[Git官网](https://git-scm.com/downloads)直接下载安装程序，建议按默认选项安装。
- 安装完成后，在开始菜单中找到“Git Bash”，打开后弹出一个类似命令行窗口的页面，即说明安装成功。
- 安装成功后，需要进行设置姓名及邮箱地址，因为Git是分布式版本控制系统，每次上传时必须记录姓名和邮箱地址。

```python
git config --global user.name "Your Name"
git config --global user.email "email@example.com"
```
> 注意git config命令的--global参数，表示这台机器上所有的Git仓库都会使用这个配置，当然也可以对某个仓库指定不同的用户名和Email地址

在Linux安装、在Mac上安装参见完整版[Git教程](https://www.liaoxuefeng.com/wiki/896043488029600
)
## 创建版本库
版本库又名仓库（repository），可以简单理解成一个目录，这个目录里面的所有文件都可以被Git管理起来，每个文件的修改、删除，Git都能跟踪，以便任何时刻都可以追踪历史，或者在将来某个时刻可以还原。

- **在合适的位置创建一个空目录**

```python
mkdir learngit
cd learngit
pwd
```
	/Users/michael/learngit

> mkdir创建新文件夹（命名为learngit），cd打开文件夹，pwd显示当前目录

- **通过git init命令把这个目录变成Git可以管理的仓库**
```python
git init
```
	Initialized empty Git repository in /Users/michael/learngit/.git/

Git已经将仓库建好，并告诉你是一个空的仓库（empty Git repository），当前目录下多了一个.git的目录（如没有看到.git目录，是因为该目录默认为隐藏的，用ls -ah命令即可看到），这个目录是Git来跟踪管理版本库的，千万不要手动修改这个目录里面的文件，不然可能会破坏Git仓库。

- **把文件添加到版本库**

所有的版本控制系统，其实只能跟踪文本文件的改动，比如TXT文件、网页、所有的程序代码等等，Git也不例外。而图片、视频、Word这些二进制文件，虽然也能由版本控制系统管理，但没法跟踪文件的变化，只能把二进制文件每次改动串起来，也就是只知道图片从100KB改成了120KB，但到底改了什么，版本控制系统不知道。

因为文本是有编码的，比如中文有常用的GBK编码，日文有Shift_JIS编码，如果没有历史遗留问题，强烈建议使用标准的UTF-8编码，所有语言使用同一种编码，既没有冲突，又被所有平台所支持。另外，建议使用Notepad++代替记事本编辑文本文件，并将其默认编码设置为UTF-8 without BOM。






