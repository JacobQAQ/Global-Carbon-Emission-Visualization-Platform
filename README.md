# Global Carbon Emission Visualization Platform

全球碳排放与气候责任交互式数据看板（1750–2024，含 2025–2035 简单趋势预测）。

## 在线访问

GitHub Pages 发布后，可通过以下地址访问 Web 看板：

https://jacobqaq.github.io/Global-Carbon-Emission-Visualization-Platform/

如果页面刚刚推送，GitHub Actions 通常需要几十秒到数分钟完成部署。

## 项目内容

- `index.html`：可直接通过 GitHub Pages 打开的静态看板页面。
- `co2_dashboard/build_dashboard.py`：生成看板的 Python 脚本。
- `co2_dashboard/output/co2_dashboard.html`：脚本生成的完整静态 HTML 看板。
- `co2_dashboard/requirements.txt`：Python 依赖。
- `.github/workflows/pages.yml`：GitHub Pages 自动部署工作流。

## 本地运行

原始数据文件 `owid-co2-data.csv` 体积较大，未提交到仓库。请从 OWID CO2 dataset 获取数据后，将其放在项目根目录或 `co2_dashboard` 上一级目录，然后运行：

```bash
cd co2_dashboard
python -m pip install -r requirements.txt
python build_dashboard.py
```

也可以显式指定数据路径：

```bash
python build_dashboard.py --data ../owid-co2-data.csv --output output/co2_dashboard.html
```

预测终点年份默认是 2035，可修改：

```bash
python build_dashboard.py --forecast-end 2040
```

## 数据与预测说明

- 数据来源：Our World in Data CO2 dataset。
- 国家地图和国家排名排除 `iso_code` 为空或以 `OWID_` 开头的汇总实体。
- GDP 历史数据主要截至 2022 年，`trade_co2` 历史数据截至 2023 年。
- 2025 年后的数值为基于历史序列的简单回归外推，仅用于课程展示和趋势探索，不代表正式气候情景、政策情景或 IPCC 预测。
- 能源来源分解存在统计覆盖差异，空值不等于零。
