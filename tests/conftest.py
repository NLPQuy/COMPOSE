import pytest

from tests.synthetic import make_cfst, make_store


@pytest.fixture(scope="session")
def data(tmp_path_factory):
    root = tmp_path_factory.mktemp("data")
    cfst_root = make_cfst(root / "CFST")
    stores = {ds: str(make_store(cfst_root, ds, root / "slots" / f"{ds}_slots_K7.pt")) for ds in ("cgqa", "cobj")}
    return cfst_root, stores
