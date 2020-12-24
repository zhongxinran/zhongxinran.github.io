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