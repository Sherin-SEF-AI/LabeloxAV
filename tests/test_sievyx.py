"""SIEVYX composition tests: the queue-composition aggregator ranks by combined priority, splits by rarity
band, and reports the class mix, so the dashboard truthfully shows what the label budget is buying."""

from services.sievyx.composition import compose


def _item(cls, value, rarity, **sig):
    return {"class_name": cls, "value": value, "scores": {"rarity": rarity, "uncertainty": sig.get("u", 0.0),
            "diversity": sig.get("d", 0.0), "error_prone": 0.0, "fn": 0.0}}


def test_empty():
    c = compose([])
    assert c["n"] == 0 and c["by_class"] == []


def test_ranks_and_truncates_by_value():
    items = [_item("cattle", 0.9, 0.9), _item("sedan", 0.2, 0.1), _item("rider", 0.5, 0.7)]
    c = compose(items, top_n=2)
    assert c["n"] == 2
    # the two highest-value items are cattle (0.9) and rider (0.5); sedan (0.2) is dropped
    names = {b["class_name"] for b in c["by_class"]}
    assert names == {"cattle", "rider"}


def test_rarity_band_split():
    items = [_item("a", 0.8, 0.9), _item("b", 0.7, 0.5), _item("c", 0.6, 0.1)]
    c = compose(items)
    assert c["by_rarity_band"]["high"]["count"] == 1
    assert c["by_rarity_band"]["medium"]["count"] == 1
    assert c["by_rarity_band"]["low"]["count"] == 1


def test_class_mix_and_shares():
    items = [_item("cattle", 0.9, 0.9), _item("cattle", 0.8, 0.8), _item("sedan", 0.5, 0.2)]
    c = compose(items)
    top = c["by_class"][0]
    assert top["class_name"] == "cattle" and top["count"] == 2 and top["share"] == round(2 / 3, 4)
    assert abs(sum(b["share"] for b in c["by_class"]) - 1.0) < 1e-6


def test_mean_signals_reported():
    items = [_item("a", 0.9, 0.9, u=0.8), _item("b", 0.7, 0.5, u=0.4)]
    c = compose(items)
    assert abs(c["mean_signals"]["uncertainty"] - 0.6) < 1e-6
    assert abs(c["mean_signals"]["rarity"] - 0.7) < 1e-6
