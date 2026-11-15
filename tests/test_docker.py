"""Guards on the container setup (D-052): secrets and the study's cache stay out of the
image, and the assistant is open to this machine only."""

from support_rag import config


def _dockerignore():
    return (config.ROOT / ".dockerignore").read_text(encoding="utf-8").splitlines()


def test_the_docker_build_never_sees_the_env_file():
    assert ".env" in _dockerignore()


def test_the_docker_build_leaves_out_the_study_cache_and_runs():
    assert {"cache", "runs"} <= set(_dockerignore())


def test_the_assistant_is_open_to_this_machine_only():
    compose = (config.ROOT / "compose.yaml").read_text(encoding="utf-8")
    assert '"127.0.0.1:8000:8000"' in compose
    assert '"127.0.0.1:5432:5432"' in compose
