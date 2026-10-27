import os

import pytest

from inferencedeck.drafts import Draft
from inferencedeck.launcher import LaunchError, LaunchPlan, plan_launch, run_plan


def draft(kind, body, working_dir=None):
    return Draft(1, "d", kind, body, working_dir, "", None, "", "")


def test_run_plan_streams_output_and_exit_code():
    lines = []
    procs = []
    code = run_plan(LaunchPlan(["/bin/sh", "-c", "echo one; echo two >&2; exit 3"], None, []), lines.append, procs.append)
    assert code == 3 and lines == ["one", "two"] and len(procs) == 1


def test_compose_plan_writes_temp_file_and_cleans_up(tmp_path):
    plan = plan_launch(draft("compose", "services: {}\n", str(tmp_path)), compose_cmd=["docker", "compose"])
    compose_file = plan.argv[plan.argv.index("-f") + 1]
    assert open(compose_file).read() == "services: {}\n"
    assert plan.argv[-2:] == ["up", "-d"] and plan.cwd == str(tmp_path)
    assert plan.argv[plan.argv.index("--project-directory") + 1] == str(tmp_path)
    plan.cleanup()
    assert not os.path.exists(compose_file)


def test_run_plan_warnings(monkeypatch):
    monkeypatch.setattr("inferencedeck.launcher.shutil.which", lambda name: "/usr/bin/docker")
    plan = plan_launch(draft("run", "docker run --name x img"))
    assert plan.argv[:2] == ["/bin/sh", "-c"]
    assert any("detach" in w for w in plan.warnings)
    assert any("does not look like" in w for w in plan_launch(draft("run", "echo hi")).warnings)
    with pytest.raises(LaunchError):
        plan_launch(draft("run", "   "))


def test_missing_docker(monkeypatch):
    monkeypatch.setattr("inferencedeck.launcher.shutil.which", lambda name: None)
    with pytest.raises(LaunchError, match="docker CLI"):
        plan_launch(draft("run", "docker run -d img"))
