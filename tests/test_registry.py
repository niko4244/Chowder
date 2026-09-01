from chowder.models import Experiment, ExperimentResult, Hypothesis
from chowder.provenance import EvidenceManifest
from chowder.registry import RunRegistry


def test_registry_persists_result_manifest_and_lineage(tmp_path):
    db = tmp_path / "runs.db"
    root = Experiment("root", None, Hypothesis("o", "c", "i"), {}, 1)
    child = Experiment("child", "root", Hypothesis("o2", "c2", "i2"), {"lr": 1e-5}, 1)
    result = ExperimentResult("child", {"score": 2.0}, 0.8, "adapter://child")
    manifest = EvidenceManifest("child", "root", {"lr": 1e-5}, "abc", {"score": 2.0}, 42)

    with RunRegistry(db) as registry:
        registry.record_experiment(root)
        registry.record_experiment(child)
        registry.record_result(result)
        digest = registry.record_manifest(manifest)
        assert registry.lineage("child") == ("root",)
        assert list(registry.list_results()) == [result]
        assert len(digest) == 64
