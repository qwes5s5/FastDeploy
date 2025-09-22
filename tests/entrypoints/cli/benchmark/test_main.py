# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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

import argparse
import unittest
from unittest.mock import MagicMock, patch

from fastdeploy.entrypoints.cli.benchmark.main import BenchmarkSubcommand, cmd_init


class MockBenchmarkSubcommand:
    """Mock subclass of BenchmarkSubcommandBase"""

    name = "mock"
    help = "Mock benchmark command"

    @classmethod
    def add_cli_args(cls, parser):
        parser.add_argument("--mock-arg", help="Mock argument")

    @staticmethod
    def cmd(args):
        return "mock_result"


class TestBenchmarkMain(unittest.TestCase):
    """Test cases for benchmark/main.py"""

    def setUp(self):
        self.subparsers = MagicMock(spec=argparse._SubParsersAction)
        self.bench_cmd = BenchmarkSubcommand()

        # Patch subclasses to include our mock
        self.patcher = patch(
            "fastdeploy.entrypoints.cli.benchmark.main.BenchmarkSubcommandBase.__subclasses__",
            return_value=[MockBenchmarkSubcommand],
        )
        self.mock_subclasses = self.patcher.start()

    def tearDown(self):
        self.patcher.stop()

    def test_subparser_init(self):
        """Test subparser initialization with mock subcommands"""
        mock_parser = MagicMock()
        mock_subparser = MagicMock()
        self.subparsers.add_parser.return_value = mock_parser
        mock_parser.add_subparsers.return_value = mock_subparser

        result = self.bench_cmd.subparser_init(self.subparsers)

        # Verify parser creation
        self.subparsers.add_parser.assert_called_once_with(
            "bench",
            help="FastDeploy bench subcommand.",
            description="FastDeploy bench subcommand.",
            usage="FastDeploy bench <bench_type> [options]",
        )

        # Verify subparsers setup
        mock_parser.add_subparsers.assert_called_once_with(required=True, dest="bench_type")

        # Verify mock subcommand registration
        mock_subparser.add_parser.assert_called_once_with(
            "mock",
            help="Mock benchmark command",
            description="Mock benchmark command",
            usage="FastDeploy bench mock [options]",
        )

        self.assertEqual(result, mock_parser)

    def test_cmd_dispatch(self):
        """Test command dispatch mechanism"""
        args = MagicMock()
        args.bench_type = "mock"
        args.dispatch_function = MockBenchmarkSubcommand.cmd

        result = self.bench_cmd.cmd(args)
        self.assertEqual(result, None)

    def test_cmd_init(self):
        """Test command initialization"""
        commands = cmd_init()
        self.assertEqual(len(commands), 1)
        self.assertIsInstance(commands[0], BenchmarkSubcommand)

    def test_validate(self):
        """Test validate method (should do nothing)"""
        args = MagicMock()

        # 测试validate方法被调用且不改变参数
        original_args = vars(args).copy()
        self.bench_cmd.validate(args)
        self.assertEqual(vars(args), original_args)

        # 测试validate方法可以被mock
        with patch.object(self.bench_cmd, "validate") as mock_validate:
            self.bench_cmd.validate(args)
            mock_validate.assert_called_once_with(args)


if __name__ == "__main__":
    unittest.main()
