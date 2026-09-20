import pytest

from causalsight.models.qwen_vl import resolve_init_adapters


def mk(d):
    d.mkdir(parents=True)
    (d / "adapter_config.json").write_text("{}")
    return d


def test_resolution_order(tmp_path):
    s1 = mk(tmp_path / "runs" / "stage1" / "adapter")
    s2 = mk(tmp_path / "runs" / "stage2_opt" / "adapter")
    assert resolve_init_adapters(s1) == []  # no config.yaml
    (s2.parent / "config.yaml").write_text("init_adapter: /content/drive/MyDrive/causalsight/runs/stage1/adapter\n")
    assert resolve_init_adapters(s2) == [s1]  # Colab path missing here -> sibling run of the same name
    assert resolve_init_adapters(s2, explicit=str(s1)) == [s1]
    (s1 / "adapter_config.json").unlink()
    with pytest.raises(FileNotFoundError):
        resolve_init_adapters(s2)
