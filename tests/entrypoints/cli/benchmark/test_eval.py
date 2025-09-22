"""
# Copyright (c) 2025  PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""

import sys
import unittest
from unittest.mock import MagicMock, patch

# 添加模块搜索路径
sys.path.insert(0, "/root/paddlejob/wenlei/FastDeploy")

from fastdeploy.entrypoints.cli.benchmark.eval import BenchmarkEvalSubcommand


class TestBenchmarkEvalCmd(unittest.TestCase):
    @patch("subprocess.run")
    def test_cmd_with_specified_args(self, mock_run):
        """测试带指定参数的cmd函数"""
        # 创建模拟的args对象
        args = MagicMock()
        args.model = "local-completions"
        args.model_args = {
            "pretrained": "/root/paddlejob/wenlei/models/ERNIE-4.5-0.3B-Paddle",
            "base_url": "http://0.0.0.0:8891/v1/completions",
            "tokenizer": "/root/paddlejob/wenlei/models/ERNIE-4.5-0.3B-Paddle",
            "trust_remote_code": True,
        }
        args.tasks = "ceval-valid_accountant"
        args.write_out = True
        args.log_samples = False
        args.arguments = []  # 其他未指定的参数

        # 设置模拟返回值
        mock_run.return_value = MagicMock(returncode=0)

        # 调用被测函数
        BenchmarkEvalSubcommand.cmd(args)

        # 验证subprocess.run被调用
        mock_run.assert_called_once()

        # 获取实际执行的命令
        actual_cmd = mock_run.call_args[0][0]

        # 验证命令基本结构
        self.assertEqual(actual_cmd[0], "lm-eval")

        # 验证关键参数
        self.assertIn("--model", actual_cmd)
        self.assertEqual(actual_cmd[actual_cmd.index("--model") + 1], "local-completions")

        self.assertIn("--model_args", actual_cmd)
        model_args = actual_cmd[actual_cmd.index("--model_args") + 1]
        self.assertIn("pretrained=/root/paddlejob/wenlei/models/ERNIE-4.5-0.3B-Paddle", model_args)
        self.assertIn("base_url=http://0.0.0.0:8891/v1/completions", model_args)
        self.assertIn("tokenizer=/root/paddlejob/wenlei/models/ERNIE-4.5-0.3B-Paddle", model_args)
        self.assertIn("trust_remote_code=True", model_args)

        self.assertIn("--tasks", actual_cmd)
        self.assertEqual(actual_cmd[actual_cmd.index("--tasks") + 1], "ceval-valid_accountant")

        self.assertIn("--write_out", actual_cmd)
        self.assertNotIn("--log_samples", actual_cmd)


if __name__ == "__main__":
    unittest.main()
