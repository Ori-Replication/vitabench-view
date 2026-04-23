# Vita Trace Viewer (GitHub Pages)

静态只读 Trace 查看页，适用于 GitHub Pages（方式 A：Deploy from a branch）。

## 目录说明

- `index.html`：页面入口
- `manifest.json`：内置 simulation 文件列表
- `data/`：可选，放脱敏后的 simulation `.json` 文件

## 使用方式

1. 直接访问页面后，本地上传 simulation `.json` 文件查看；
2. 或把文件放到 `data/` 并更新 `manifest.json` 的 `simulation_files`，即可在线加载；
3. 仅支持只读 trace 展示，不包含工具调用能力。

## 安全要求

- 不要提交含 `Authorization`、`Bearer`、`sk_`、`cookie` 的日志；
- 仅发布脱敏数据；
- API token 必须通过环境变量或本地私密配置注入，禁止硬编码入库。
