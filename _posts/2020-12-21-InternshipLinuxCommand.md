---
title: 实习收获 | Linux常用命令
author: 钟欣然
date: 2020-12-21 06:44:00 +0800
categories: [实习收获, 字节西瓜]
math: true
mermaid: true
---



# 路径相关

* 前往某路径：`cd xx`
* 返回上一级菜单：`cd ..`
* 查看当前路径下的文件：`ls`
* 查看当前路径：`pwd`
* 创建目录：`mkdir`
* 创建文件：`touch xxx.txt`
* 删除文件夹及其内所有文件：`rm -rf xxx`
* 查看文件大小：`du -h --max-depth=1`



# 上传下载压缩相关

* 上传文件到服务器：`scp /local_path/filename username@servername:/remote_path  `
* 从服务器下载文件：`scp username@servername:/remote_path/filename /local_path`
* 上传文件夹到服务器：`scp -r /local_path/local_dir username@servername:/remote_path  `
* 从服务器下载文件夹：`scp -r username@servername:/remote_path/remote_dir /local_path`
* 压缩文件：
  * `tar zcvf FileName.tar.gz DirName`
* 解压缩文件：
  * `unzip filename. zip`
  * `tar -zxvf filename. tar.gz`
  * `tar -Jxvf filename. tar.xz`
  * `tar -Zxvf filename. tar.Z`



# 其他

* 打印日志：`tail -f xxx.log`
* 查看端口号/查找进程号：`lsof -i:xxxx`
* 查看历史命令：`history` 
* 后台运行：`nohup xxx &`
* 杀死进程：`kill -9 xxx ` 
* 查看编译历史：`ls -lh` 

# vim

所有的 Unix Like 系统都会内建 vi 文书编辑器，其他的文书编辑器则不一定会存在。但是目前我们使用比较多的是 vim 编辑器。vim 具有程序编辑的能力，可以主动的以字体颜色辨别语法的正确性，方便程序设计。vi/vim 分为三种模式：

* 命令模式：通过`vim xxx.txt`启动 vim 后，进入命令模式
  * `i` ：切换到输入模式
  * `:`：切换到底线命令模式
  * `x`：删除光标所在处的字符
* 输入模式：可以输入、删除字符，通过回车换行
  * `ESC`：切换到命令模式
* 底线命令格式：
  * `q`：退出文件
  * `w`：保存文件
  * `wq`：保存文件并退出

[更多 vim 命令与快捷键](https://www.runoob.com/linux/linux-vim.html)



# Git

* `git clone xxx`：拉取远程代码到本地
* `git fetch origin remote_branch_name:local_branch_name`：拉取远程分支到本地
* `git pull`：拉取远程对应分支的最新代码到本地
* `git add .`
* `git commit -m "xxx"`
* `git push origin local_branch_name:remote_branch_name` 
* `git push origin --delete remote_branch_name`：删除远程分支
* `git branch`：查看分支
* `git branch -vv`：查看本地分支与远程分支的关系
* `git branch -d local_branch_name`/`git branch -D local_branch_name`：删除本地分支
* `git checkout xxx`：切换分支
* `git checkout -b new_branch:old_branch`：克隆本地分支，保留修改
* `git branch --set-upstream-to origin/remote_branch_name`：将当前分支和远程分支关联起来
* `git status` ：查看现在的状态
* `git diff`：查看不同
* `git log`：查看提交日志
* `git reset HEAD <file>` ：撤回已经add的文件，保留修改，file参数可省略
* `git reset HEAD^` `git reset HAED^^` `git reset HEAD~10` `git reset commit_id` ：版本回退，分别为回退1个版本、2个版本、10个版本、回退到commit_id这个版本，其中commit_id可以通过`git log`查询
* `git stash` ：把修改存到缓存中
* `git stash pop` ：读出缓存
* `git ls-files`：查看git跟踪的文件和文件夹