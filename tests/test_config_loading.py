"""Config loading: successive loads must be independent, and the released config must reload."""
import yaml

from engine.core.yaml_utils import load_config
from tools.model_card.runner import write_resolved_config


class _Cfg:
    def __init__(self, yaml_cfg):
        self.yaml_cfg = yaml_cfg


def test_successive_load_config_calls_do_not_share_state(tmp_path):
    (tmp_path / "a.yml").write_text("only_in_a: 1\nshared: from_a\n")
    (tmp_path / "b.yml").write_text("shared: from_b\n")
    first = load_config(str(tmp_path / "a.yml"))
    second = load_config(str(tmp_path / "b.yml"))
    assert second == {"shared": "from_b"}
    assert first == {"only_in_a": 1, "shared": "from_a"}


def test_includes_still_merge_within_one_load(tmp_path):
    (tmp_path / "base.yml").write_text("lr: 0.1\nmodel: {depth: 3, width: 8}\n")
    (tmp_path / "child.yml").write_text("__include__: ['base.yml']\nmodel: {depth: 6}\n")
    loaded = load_config(str(tmp_path / "child.yml"))
    assert loaded["lr"] == 0.1
    assert loaded["model"] == {"depth": 6, "width": 8}


def test_resolved_config_drops_include_and_reloads_from_anywhere(tmp_path):
    (tmp_path / "base.yml").write_text("lr: 0.1\ndata: {img_folder: /home/me/data/train}\n")
    (tmp_path / "child.yml").write_text("__include__: ['base.yml']\nepoches: 5\n")
    merged = load_config(str(tmp_path / "child.yml"))
    assert "__include__" in merged  # the loader keeps it, which is why the dump must drop it

    elsewhere = tmp_path / "released"
    elsewhere.mkdir()
    write_resolved_config(_Cfg(merged), elsewhere / "config.yaml")
    text = (elsewhere / "config.yaml").read_text()
    assert "__include__" not in text and "/home/me" not in text

    reloaded = load_config(str(elsewhere / "config.yaml"))
    assert reloaded["lr"] == 0.1 and reloaded["epoches"] == 5
    assert yaml.safe_load(text)["data"]["img_folder"] == "<local>/data/train"
