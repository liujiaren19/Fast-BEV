#!/usr/bin/env python3
"""旧路径兼容入口。

主实现已经移动到 ``tools/data_converter/n7_raw_3dod_to_fastbev_pkl.py``。
保留该文件是为了让历史命令 ``python new_tool/unified_processor_raw.py``
继续可用；新命令建议直接调用 tools/data_converter 下的脚本。
"""

from pathlib import Path
import runpy

TARGET = Path(__file__).resolve().parents[1] / 'tools' / 'data_converter' / 'n7_raw_3dod_to_fastbev_pkl.py'


if __name__ == '__main__':
    runpy.run_path(str(TARGET), run_name='__main__')
