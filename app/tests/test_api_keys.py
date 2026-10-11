from beanie import PydanticObjectId

from security.api_keys import (
    KEY_PREFIX,
    ApiKeyInfo,
    auth_mode,
    display_prefix,
    extract_api_key,
    generate_api_key,
    has_scope,
    hash_api_key,
)


def test_generated_keys_are_unique_and_prefixed():
    first, second = generate_api_key(), generate_api_key()

    assert first != second
    assert first.startswith(KEY_PREFIX) and len(first) > 40
    assert display_prefix(first) == first[:12]


def test_hash_is_stable_and_does_not_contain_the_key():
    key = generate_api_key()

    assert hash_api_key(key) == hash_api_key(key)
    assert len(hash_api_key(key)) == 64
    assert key not in hash_api_key(key)


def test_extract_api_key_from_supported_headers():
    assert extract_api_key({"x-api-key": " atq_a "}) == "atq_a"
    assert extract_api_key({"authorization": "Bearer atq_b"}) == "atq_b"
    assert extract_api_key({"authorization": "bearer atq_c"}) == "atq_c"
    assert extract_api_key({"api-key": "atq_d"}) == "atq_d"
    assert extract_api_key({"authorization": "Basic abc"}) is None
    assert extract_api_key({}) is None


def test_scopes():
    info = ApiKeyInfo(
        key_id=PydanticObjectId(),
        project_id=PydanticObjectId(),
        scopes=frozenset({"ingest"}),
        name="ci",
        prefix="atq_abcdefgh",
    )

    assert has_scope(info, "ingest")
    assert not has_scope(info, "read")
    assert has_scope(info, None)


def test_auth_mode_defaults_to_enforce(monkeypatch):
    monkeypatch.delenv("AITRAJECTIQ_AUTH", raising=False)
    assert auth_mode() == "enforce"
    monkeypatch.setenv("AITRAJECTIQ_AUTH", "Permissive")
    assert auth_mode() == "permissive"
    monkeypatch.setenv("AITRAJECTIQ_AUTH", "anything-else")
    assert auth_mode() == "enforce"
