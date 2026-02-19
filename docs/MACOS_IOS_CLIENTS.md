# MacOS + iOS Clients

本项目优先补齐 MacOS（桌面壳）与 iOS（PWA）形态。

## MacOS Desktop (Electron Shell)

位置：`desktop/macos`

### 启动方式
1. 启动后端与前端（前端默认 `http://127.0.0.1:3000`）
2. 启动桌面壳：
```bash
cd ./desktop/macos
npm install
ELECTRON_START_URL=http://127.0.0.1:3000/chat npm run dev
```

说明：
- 默认只作为壳层承载现有 Console，保持与 Web 功能一致。
- Electron 已启用 `contextIsolation` + `sandbox`。

## iOS (PWA)

前端已补齐：
- `manifest.webmanifest`（由 `frontend/app/manifest.ts` 输出）
- iOS 图标路由（`frontend/app/apple-icon.tsx`）
- Service Worker（`frontend/public/sw.js`）

### iPhone 安装
1. 在 Safari 打开 Console 地址（建议通过内网 https 反代）
2. 点击“分享” -> “添加到主屏幕”
3. 主屏幕图标启动后将以独立应用形态运行

### iOS 注意事项
- PWA 安装体验建议通过 HTTPS
- iOS 后台执行与推送能力受系统限制，建议把长任务交给后端调度器执行
