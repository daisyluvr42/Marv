# Deploy On MacBook Pro M1 (2020)

## 1. 初次准备
```bash
xcode-select --install
brew install uv
```

Node 建议使用 `.nvmrc` 指定版本（20.11.1）。

## 2. 拉代码并完成环境准备
```bash
REPO_URL=https://github.com/daisyluvr42/Marv.git \
TARGET_DIR=$HOME/Marv \
BRANCH=main \
bash scripts/bootstrap_mbp_m1.sh
```

## 3. 启动后端栈（Core + Edge）
```bash
cd $HOME/Marv
bash scripts/start_stack.sh
```

前端开发模式：
```bash
cd $HOME/Marv/frontend
NEXT_PUBLIC_API_BASE_URL=http://127.0.0.1:8000 npm run dev
```

## 4. 验证
```bash
cd $HOME/Marv
EDGE_BASE_URL=http://127.0.0.1:8000 bash scripts/smoke_test.sh
```

## 5. 停止
```bash
cd $HOME/Marv
bash scripts/stop_stack.sh
```

## 6. 更新版本
```bash
cd $HOME/Marv
git pull --ff-only
bash scripts/bootstrap_mbp_m1.sh
bash scripts/start_stack.sh
```
