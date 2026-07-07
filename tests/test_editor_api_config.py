from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from src.utils import mask_secret, resolve_api_provider_config, resolve_editor_provider_config


class EditorApiConfigTests(unittest.TestCase):
    def make_config(self, provider: str = "openrouter", model: str = "") -> dict:
        return {
            "editor": {
                "provider": "api",
                "api": {
                    "provider": provider,
                    "model": model,
                    "secrets_file": "secrets.local.yaml",
                },
            }
        }

    def test_openrouter_default_model_and_masked_key(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "secrets.local.yaml").write_text(
                "api_keys:\n  openrouter: \"sk-or-test-not-real\"\n",
                encoding="utf-8",
            )
            resolved = resolve_api_provider_config(self.make_config(), project_root=root)
        self.assertEqual(resolved["provider"], "openrouter")
        self.assertEqual(resolved["resolved_model"], "google/gemini-2.5-flash-lite")
        self.assertEqual(resolved["api_key_masked"], "sk-...real")

    def test_google_default_model(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "secrets.local.yaml").write_text(
                "api_keys:\n  google: \"AQ-test-not-real\"\n",
                encoding="utf-8",
            )
            resolved = resolve_api_provider_config(self.make_config(provider="google"), project_root=root)
        self.assertEqual(resolved["provider"], "google")
        self.assertEqual(resolved["base_url"], "https://generativelanguage.googleapis.com/v1beta/openai")
        self.assertEqual(resolved["resolved_model"], "gemini-3.5-flash")
        self.assertEqual(resolved["api_key_masked"], "AQ...real")

    def test_model_override(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "secrets.local.yaml").write_text(
                "api_keys:\n  deepseek: \"sk-test-not-real\"\n",
                encoding="utf-8",
            )
            resolved = resolve_api_provider_config(self.make_config(provider="deepseek", model="deepseek-v4-flash"), project_root=root)
        self.assertEqual(resolved["resolved_model"], "deepseek-v4-flash")
        self.assertTrue(resolved["api_no_think"])

    def test_deepseek_default_model(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "secrets.local.yaml").write_text(
                "api_keys:\n  deepseek: \"sk-test-not-real\"\n",
                encoding="utf-8",
            )
            resolved = resolve_api_provider_config(self.make_config(provider="deepseek"), project_root=root)
        self.assertEqual(resolved["resolved_model"], "deepseek-v4-flash")
        self.assertTrue(resolved["api_no_think"])

    def test_top_level_api_config(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "secrets.local.yaml").write_text(
                "api_keys:\n  openrouter: \"sk-or-test-not-real\"\n",
                encoding="utf-8",
            )
            resolved = resolve_api_provider_config(
                {
                    "editor": {"provider": "api", "no_think": False},
                    "api": {"provider": "openrouter", "model": "", "secrets_file": "secrets.local.yaml"},
                },
                project_root=root,
            )
        self.assertEqual(resolved["provider"], "openrouter")
        self.assertFalse(resolved["api_no_think"])

    def test_editor_provider_preset_config(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "secrets.local.yaml").write_text(
                "api_keys:\n  dashscope: \"sk-test-not-real\"\n",
                encoding="utf-8",
            )
            resolved = resolve_editor_provider_config(
                {
                    "editor": {"provider": "dashscope", "no_think": True},
                    "secrets": {"file": "secrets.local.yaml"},
                    "editor_providers": {
                        "dashscope": {
                            "kind": "openai_compatible",
                            "base_url": "https://workspace.example.com/compatible-mode/v1",
                            "model": "qwen3-max",
                            "api_key_name": "dashscope",
                        }
                    },
                },
                project_root=root,
            )
        self.assertEqual(resolved["provider"], "dashscope")
        self.assertEqual(resolved["base_url"], "https://workspace.example.com/compatible-mode/v1")
        self.assertEqual(resolved["resolved_model"], "qwen3-max")
        self.assertTrue(resolved["api_no_think"])

    def test_missing_key_error_names_field(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "secrets.local.yaml").write_text("api_keys:\n  openrouter: \"\"\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "api_keys.openrouter"):
                resolve_api_provider_config(self.make_config(), project_root=root)

    def test_top_level_base_url_override(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "secrets.local.yaml").write_text(
                "api_keys:\n  dashscope: \"sk-test-not-real\"\n",
                encoding="utf-8",
            )
            resolved = resolve_api_provider_config(
                {
                    "editor": {"provider": "dashscope"},
                    "secrets": {"file": "secrets.local.yaml"},
                    "editor_providers": {
                        "dashscope": {
                            "kind": "openai_compatible",
                            "base_url": "https://workspace.example.com/compatible-mode/v1",
                            "model": "qwen3-max",
                            "api_key_name": "dashscope",
                        }
                    },
                },
                project_root=root,
            )
        self.assertEqual(resolved["provider"], "dashscope")
        self.assertEqual(resolved["base_url"], "https://workspace.example.com/compatible-mode/v1")
        self.assertEqual(resolved["resolved_model"], "qwen3-max")

    def test_legacy_top_level_api_config_migrates_through_load_config_shape(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "secrets.local.yaml").write_text(
                "api_keys:\n  dashscope: \"sk-test-not-real\"\n",
                encoding="utf-8",
            )
            resolved = resolve_api_provider_config(
                {
                    "editor": {"provider": "api"},
                    "secrets": {"file": "secrets.local.yaml"},
                    "editor_providers": {
                        "dashscope": {
                            "kind": "openai_compatible",
                            "base_url": "https://workspace.example.com/compatible-mode/v1",
                            "model": "qwen3-max",
                            "api_key_name": "dashscope",
                        }
                    },
                },
                project_root=root,
            )
        self.assertEqual(resolved["provider"], "dashscope")
        self.assertEqual(resolved["base_url"], "https://workspace.example.com/compatible-mode/v1")
        self.assertEqual(resolved["resolved_model"], "qwen3-max")

    def test_unknown_provider_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "secrets.local.yaml").write_text("api_keys:\n  unknown: \"sk-test-not-real\"\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "不支持的 editor.provider"):
                resolve_api_provider_config(self.make_config(provider="custom"), project_root=root)

    def test_mask_secret(self) -> None:
        self.assertEqual(mask_secret("sk-test-not-real"), "sk-...real")


if __name__ == "__main__":
    unittest.main()
