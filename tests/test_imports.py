import importlib


def test_merge_module_and_function_imports():
    import carwatch
    from carwatch import merge, merge_saliva
    from carwatch.merge import merge_saliva as merge_saliva_from_module

    assert carwatch.merge is merge
    assert carwatch.merge_saliva is merge_saliva
    assert merge.merge_saliva is merge_saliva_from_module
    assert merge_saliva is merge_saliva_from_module
    assert "merge" in carwatch.__all__
    assert "merge_saliva" in carwatch.__all__


def test_merge_import_with_cached_pre_merge_exceptions(monkeypatch):
    import carwatch.exceptions as exceptions
    import carwatch.merge as merge

    monkeypatch.delattr(exceptions, "MergeError")

    reloaded = importlib.reload(merge)

    assert callable(reloaded.merge_saliva)
