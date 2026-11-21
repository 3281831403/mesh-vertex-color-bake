# mesh-vertex-color-bake

## 题面

把 AO 烘到顶点色后，模型整体发黑、边缘还渗色：顶点法线做了等权平均，小面反而主导大面的朝向；半球采样没有按法线定向，光线整片朝下打；遮挡射线把起点也算作命中，表面自己把自己挡成全黑；方向光项没有钳制，背光顶点拿到负亮度；烘焙结果没有归一，颜色一路冲过 1；邻接平滑把边界顶点也卷进平均，颜色渗出模型边缘；颜色越界不校验，导出色块被硬裁；顶点数与颜色数不符也不报错，索引整体错位；批量烘焙共用同一份颜色数组，网格之间互相串色；`samples=0` 不校验，随后直接除零。vcolor/bake.py 是核心，tests/test_bake.py 15个红。要修的是：上面十处缺陷逐条改正，只改实现、不动 tests，让 15 个用例全部转绿。

## 运行方式

在仓库根目录执行：

```powershell
python -m pytest tests -q
```

初始仓库是失败的 Bugfix 环境：恰好 `15 failed`，不能出现 collection error、import error 或 syntax error。修复 `vcolor/bake.py` 后 15 条测试应全部通过。

## 字段表

| 字段 | 取值 |
| --- | --- |
| 所属领域 | 游戏、图形与多媒体 |
| 子方向 | 图形渲染 |
| 任务类型 | Bugfix |
| 语言/框架 | Python 3, pytest |
| 环境可复现等级 | 无外部依赖 |

## tests 不可修改

`tests/test_bake.py` 是验收标准，不可修改、不可删除、不可跳过、不可调低断言强度。修复只能落在 `vcolor/bake.py`（必要时可调整 `vcolor/__init__.py`）。

## 约束

- 只使用 Python 标准库（`math`）与 `pytest`；离线运行，不使用随机数。
- 颜色通道范围 `0..1`；AO 范围 `0..1`（1 表示完全敞开，0 表示完全遮挡）；数值比较容差 `1e-6`。
- 网格字典字段为 `positions`、`normals`、`indices`。
- 非法输入统一抛 `VColorError`。
