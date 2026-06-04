下面给你一份可以直接沉淀到 Obsidian 的 **GitHub + Git + SSH 从零配置全流程**。适合你这种场景：Mac 本地 / 远程服务器 / VS Code Remote SSH 都要能正常拉代码、提交、推送。

---

# GitHub 配置全流程

## 0. 先明确几个概念

Git 是本地版本管理工具。

GitHub 是远程代码仓库平台。

SSH key 是你电脑/服务器和 GitHub 之间的身份凭证。

`.git` 是项目本地 Git 仓库数据目录。

`origin` 是远程仓库的默认名字，通常指向 GitHub。

---

# 1. 检查 Git 是否安装

在终端输入：

```bash
git --version
```

如果能看到类似：

```bash
git version 2.xx.x
```

说明 Git 已经安装。

---

# 2. 配置 Git 提交身份

这个身份会写进 commit 记录里。

```bash
git config --global user.name "你的GitHub用户名或名字"
git config --global user.email "你的GitHub邮箱"
```

例如：

```bash
git config --global user.name "jiaqiang000"
git config --global user.email "你的邮箱@example.com"
```

查看是否配置成功：

```bash
git config --global --list
```

应该能看到：

```bash
user.name=xxx
user.email=xxx
```

注意：  
这个不是登录 GitHub，只是设置 commit 作者信息。

---

# 3. 生成 SSH key

先查看有没有已有 SSH key：

```bash
ls ~/.ssh
```

如果看到类似：

```bash
id_rsa
id_rsa.pub
id_ed25519
id_ed25519.pub
```

说明以前生成过。

如果没有，推荐生成新的 ed25519 key：

```bash
ssh-keygen -t ed25519 -C "你的GitHub邮箱"
```

一路回车即可。

默认会生成：

```bash
~/.ssh/id_ed25519
~/.ssh/id_ed25519.pub
```

其中：

```bash
id_ed25519
```

是私钥，不能泄露。

```bash
id_ed25519.pub
```

是公钥，可以放到 GitHub。

---

# 4. 把公钥添加到 GitHub

查看公钥内容：

```bash
cat ~/.ssh/id_ed25519.pub
```

复制整行内容，通常长这样：

```bash
ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIxxxxxxx your_email@example.com
```

然后去 GitHub：

```text
GitHub 头像
→ Settings
→ SSH and GPG keys
→ New SSH key
```

Title 随便写，比如：

```text
MacBook Pro
```

Key 粘贴刚才的公钥。

保存。

---

# 5. 测试 SSH 是否连通 GitHub

执行：

```bash
ssh -T git@github.com
```

第一次可能会问：

```text
Are you sure you want to continue connecting?
```

输入：

```bash
yes
```

成功的话会看到类似：

```text
Hi jiaqiang000! You've successfully authenticated, but GitHub does not provide shell access.
```

这句话表示 GitHub SSH 配置成功。

注意：  
它说 `does not provide shell access` 是正常的，不是报错。意思是 GitHub 只允许你用 SSH 拉代码/推代码，不允许你像登录服务器一样进入 GitHub 终端。

---

# 6. 配置多个 GitHub 身份时使用 ~/.ssh/config

如果你只有一个 GitHub 账号，其实可以不用配。

如果你有多个 key，建议配置：

```bash
nano ~/.ssh/config
```

加入：

```sshconfig
Host github.com
  HostName github.com
  User git
  IdentityFile ~/.ssh/id_ed25519
  IdentitiesOnly yes
```

保存后测试：

```bash
ssh -T git@github.com
```

如果你有多个 GitHub 账号，可以这样：

```sshconfig
Host github-jiaqiang
  HostName github.com
  User git
  IdentityFile ~/.ssh/id_ed25519_jiaqiang
  IdentitiesOnly yes
```

那么 clone 地址就要从：

```bash
git@github.com:jiaqiang000/repo.git
```

改成：

```bash
git@github-jiaqiang:jiaqiang000/repo.git
```

---

# 7. 克隆 GitHub 仓库

进入你想放项目的目录：

```bash
cd ~/IdeaProjects
```

使用 SSH 地址 clone：

```bash
git clone git@github.com:用户名/仓库名.git
```

例如：

```bash
git clone git@github.com:jiaqiang000/let-it-go-fork.git
```

进入项目：

```bash
cd let-it-go-fork
```

查看远程仓库：

```bash
git remote -v
```

正常会看到：

```bash
origin git@github.com:jiaqiang000/let-it-go-fork.git (fetch)
origin git@github.com:jiaqiang000/let-it-go-fork.git (push)
```

---

# 8. 如果本地已经有项目，想绑定 GitHub 仓库

进入项目目录：

```bash
cd /path/to/your/project
```

初始化 Git：

```bash
git init
```

添加远程仓库：

```bash
git remote add origin git@github.com:用户名/仓库名.git
```

查看：

```bash
git remote -v
```

---

# 9. 添加 .gitignore

项目根目录创建：

```bash
touch .gitignore
```

常见内容：

```gitignore
# macOS
.DS_Store

# Python
__pycache__/
*.pyc
.venv/
venv/
.env

# IDE
.idea/
.vscode/

# logs
*.log

# data / output
data/
outputs/
checkpoints/
```

如果是 Obsidian 笔记，最少建议：

```gitignore
.DS_Store
```

如果某个文件已经被 Git 跟踪了，再加 `.gitignore` 不会自动取消，需要：

```bash
git rm --cached 文件名
```

例如：

```bash
git rm --cached .DS_Store
```

然后提交：

```bash
git add .gitignore
git commit -m "add gitignore"
```

---

# 10. 第一次提交代码

查看状态：

```bash
git status
```

添加所有文件：

```bash
git add .
```

提交：

```bash
git commit -m "init project"
```

推送到 GitHub：

```bash
git push -u origin main
```

如果你的分支叫 master：

```bash
git push -u origin master
```

查看当前分支：

```bash
git branch
```

---

# 11. 日常使用流程

以后每天最常用的就是这几步：

```bash
git status
git add .
git commit -m "说明这次改了什么"
git push
```

例如：

```bash
git status
git add .
git commit -m "update experiment scripts"
git push
```

---

# 12. 拉取远程更新

如果别人或者其他机器改了代码，先拉：

```bash
git pull
```

如果出现你之前遇到的提示：

```text
You have divergent branches and need to specify how to reconcile them.
```

意思是：本地和远程都有新提交，Git 不知道该用 merge 还是 rebase。

推荐你个人项目使用 rebase：

```bash
git pull --rebase origin main
```

你之前这个命令成功了：

```bash
git pull --rebase origin main
```

说明本地提交已经被重新接到远程最新代码后面了。

也可以全局设置默认 pull 用 rebase：

```bash
git config --global pull.rebase true
```

以后直接：

```bash
git pull
```

默认就是 rebase。

---

# 13. 远程服务器也要单独配置 SSH key

重点：  
你的 Mac 和远程服务器是两台不同机器。

Mac 上配置了 GitHub SSH，不代表服务器也能 push/pull。

在服务器上也要执行一遍：

```bash
ssh-keygen -t ed25519 -C "你的GitHub邮箱"
cat ~/.ssh/id_ed25519.pub
```

把服务器的公钥也添加到 GitHub。

然后服务器上测试：

```bash
ssh -T git@github.com
```

成功后服务器才可以：

```bash
git clone
git pull
git push
```

---

# 14. VS Code Remote SSH 和 GitHub SSH 的关系

它们是两套 SSH。

## VS Code Remote SSH

用于你的 Mac 连接服务器：

```bash
ssh i-2.gpushare.com
```

配置通常在 Mac 的：

```bash
~/.ssh/config
```

例如：

```sshconfig
Host i-2.gpushare.com
  HostName i-2.gpushare.com
  User root
  Port 22
  IdentityFile ~/.ssh/你的服务器私钥
```

## GitHub SSH

用于 Mac 或服务器连接 GitHub：

```bash
ssh -T git@github.com
```

配置一般是：

```sshconfig
Host github.com
  HostName github.com
  User git
  IdentityFile ~/.ssh/id_ed25519
  IdentitiesOnly yes
```

所以：

```text
Mac → 服务器
```

和：

```text
Mac/服务器 → GitHub
```

不是一回事。

---

# 15. 常见问题解释

## 15.1 Your branch is ahead of 'origin/main' by 2 commits

意思是：你本地比 GitHub 远程多 2 个 commit。

解决：

```bash
git push
```

推上去后就同步了。

## 15.2 nothing to commit, working tree clean

意思是：当前没有未提交的修改。

这是正常状态。

## 15.3 Everything up-to-date

意思是：远程已经是最新了，没有东西需要 push。

## 15.4 Permission denied publickey

意思是 SSH key 没配好。

检查：

```bash
ssh -T git@github.com
```

然后检查：

```bash
ls ~/.ssh
cat ~/.ssh/config
```

## 15.5 remote origin already exists

意思是已经有 origin 了。

查看：

```bash
git remote -v
```

修改 origin：

```bash
git remote set-url origin git@github.com:用户名/仓库名.git
```

删除 origin：

```bash
git remote remove origin
```

然后重新添加：

```bash
git remote add origin git@github.com:用户名/仓库名.git
```

---

# 16. 推荐你的最终配置习惯

Mac 本地配置：

```bash
git config --global user.name "jiaqiang000"
git config --global user.email "你的GitHub邮箱"
git config --global pull.rebase true
```

服务器也配置：

```bash
git config --global user.name "jiaqiang000"
git config --global user.email "你的GitHub邮箱"
git config --global pull.rebase true
```

日常项目操作：

```bash
git pull --rebase
git status
git add .
git commit -m "清楚描述修改内容"
git push
```

---

# 17. 一套最小可用命令总结

新机器从零配置：

```bash
git config --global user.name "jiaqiang000"
git config --global user.email "你的GitHub邮箱"
git config --global pull.rebase true

ssh-keygen -t ed25519 -C "你的GitHub邮箱"
cat ~/.ssh/id_ed25519.pub
```

把公钥加到 GitHub 后测试：

```bash
ssh -T git@github.com
```

克隆项目：

```bash
git clone git@github.com:jiaqiang000/仓库名.git
cd 仓库名
```

日常提交：

```bash
git status
git add .
git commit -m "update"
git push
```

日常拉取：

```bash
git pull --rebase
```

---

你可以把这个标题叫：

```text
GitHub / Git / SSH 从零配置流程
```