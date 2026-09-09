from app.trace import runtime


def test_host_awake_guard_is_an_explicit_noop_off_windows(monkeypatch):
    monkeypatch.setattr(runtime.os, 'name', 'posix')
    guard=runtime.HostAwakeGuard(enabled=True)
    assert guard.acquire()=={
        'requested':True,'active':False,'platform':'posix'}
    guard.release()
    assert guard.active is False


def test_disabled_host_awake_guard_never_calls_platform_api(monkeypatch):
    monkeypatch.setattr(runtime.os, 'name', 'nt')
    guard=runtime.HostAwakeGuard(enabled=False)
    assert guard.acquire()['active'] is False
    guard.release()
