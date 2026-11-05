"""OpenAPI of POST /v1/jobs: upload fields are file pickers in /docs (format binary)."""


def _create_job_body(schema: dict) -> dict:
    content = schema["paths"]["/v1/jobs"]["post"]["requestBody"]["content"]["multipart/form-data"]
    name = content["schema"]["$ref"].rsplit("/", 1)[-1]
    return schema["components"]["schemas"][name]


def test_files_fields_are_binary(api):
    body = _create_job_body(api.client.get("/openapi.json").json())
    for field in ("files", "files[]"):
        prop = body["properties"][field]
        assert prop["type"] == "array", prop
        assert prop["items"]["type"] == "string"
        assert prop["items"]["format"] == "binary"
        # Swagger UI sends an empty part (ignored) for an untouched field, not its sample "string".
        assert prop["items"]["example"] == ""
        assert "anyOf" not in prop
        assert prop["description"]
    assert body["properties"]["options"]["anyOf"][0]["type"] == "string"  # other fields unchanged


def test_schema_is_stable_across_requests(api):
    first = api.client.get("/openapi.json").json()
    assert api.client.get("/openapi.json").json() == first
