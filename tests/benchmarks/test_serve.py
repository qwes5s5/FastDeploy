import argparse
import unittest
from unittest.mock import patch

from fastdeploy.benchmarks.serve import main


class TestServeMain(unittest.TestCase):
    def setUp(self):
        # 创建模拟的args对象
        self.args = argparse.Namespace(
            backend="vllm",
            model="test-model",
            tokenizer=None,
            dataset_name="EB",
            dataset_path=None,
            num_prompts=1,
            request_rate=1.0,
            burstiness=1.0,
            max_concurrency=None,
            disable_tqdm=True,
            profile=False,
            percentile_metrics="ttft,tpot,e2el",
            metric_percentiles="50,90,99",
            ignore_eos=False,
            debug=False,
            goodput=None,
            save_result=False,
            save_detailed=False,
            result_dir=None,
            result_filename=None,
            metadata=None,
            hyperparameter_path=None,
            sharegpt_output_len=None,
            random_input_len=1024,
            random_output_len=128,
            random_range_ratio=0.0,
            random_prefix_len=0,
            hf_subset=None,
            hf_split=None,
            hf_output_len=None,
            top_p=None,
            top_k=None,
            min_p=None,
            temperature=None,
            tokenizer_mode="auto",
            served_model_name=None,
            lora_modules=None,
            base_url=None,
            host="127.0.0.1",
            port=8000,
            endpoint="/v1/completions",
            logprobs=None,
        )

    @patch("fastdeploy.benchmarks.serve.asyncio.run")
    @patch("fastdeploy.benchmarks.serve.EBDataset")
    def test_main_with_eb_dataset(self, mock_dataset, mock_run):
        """测试使用EBDataset时的main函数"""
        # 设置mock
        mock_dataset.return_value.sample.return_value = []
        mock_run.return_value = {"status": "success"}

        # 调用main函数
        main(self.args)

        # 验证
        mock_dataset.assert_called_once()
        mock_run.assert_called_once()

    @patch("fastdeploy.benchmarks.serve.asyncio.run")
    @patch("fastdeploy.benchmarks.serve.EBChatDataset")
    def test_main_with_ebchat_dataset(self, mock_dataset, mock_run):
        """测试使用EBChatDataset时的main函数"""
        # 修改args
        self.args.dataset_name = "EBChat"

        # 设置mock
        mock_dataset.return_value.sample.return_value = []
        mock_run.return_value = {"status": "success"}

        # 调用main函数
        main(self.args)

        # 验证
        mock_dataset.assert_called_once()
        mock_run.assert_called_once()

    @patch("fastdeploy.benchmarks.serve.asyncio.run")
    @patch("fastdeploy.benchmarks.serve.EBDataset")
    def test_main_with_save_result(self, mock_dataset, mock_run):
        """测试保存结果时的main函数"""
        # 修改args
        self.args.save_result = True
        self.args.result_dir = "/tmp/results"

        # 设置mock
        mock_dataset.return_value.sample.return_value = []
        mock_run.return_value = {"status": "success"}

        # 调用main函数
        main(self.args)

        # 验证
        mock_dataset.assert_called_once()
        mock_run.assert_called_once()

    @patch("fastdeploy.benchmarks.serve.asyncio.run")
    @patch("fastdeploy.benchmarks.serve.EBDataset")
    def test_main_with_lora_modules(self, mock_dataset, mock_run):
        """测试使用LoRA模块时的main函数"""
        # 修改args
        self.args.lora_modules = ["lora1", "lora2"]

        # 设置mock
        mock_dataset.return_value.sample.return_value = []
        mock_run.return_value = {"status": "success"}

        # 调用main函数
        main(self.args)

        # 验证
        mock_dataset.assert_called_once()
        mock_run.assert_called_once()

    @patch("fastdeploy.benchmarks.serve.asyncio.run")
    @patch("fastdeploy.benchmarks.serve.EBDataset")
    def test_main_with_hyperparameters(self, mock_dataset, mock_run):
        """测试使用超参数文件时的main函数"""
        # 修改args
        self.args.hyperparameter_path = "test_config.yaml"

        # 设置mock
        mock_dataset.return_value.sample.return_value = []
        mock_run.return_value = {"status": "success"}

        # 模拟yaml文件读取
        with patch("fastdeploy.benchmarks.serve.yaml.safe_load", return_value={"param1": "value1"}):
            # 调用main函数
            main(self.args)

            # 验证
            mock_dataset.assert_called_once()
            mock_run.assert_called_once()


if __name__ == "__main__":
    unittest.main()
