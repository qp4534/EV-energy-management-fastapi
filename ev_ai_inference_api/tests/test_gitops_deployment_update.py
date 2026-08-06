from pathlib import Path

from scripts.update_gitops_fastapi_deployment import update_manifest


BASE_MANIFEST = """\
apiVersion: apps/v1
kind: Deployment
spec:
  template:
    spec:
      containers:
        - name: api
          image: qp4534/ev-energy-management-fastapi:old
          env:
            - name: DATABASE_URL
              value: test
          resources:
            requests:
              cpu: 250m
              memory: 768Mi
            limits:
              cpu: "1"
              memory: 1536Mi
"""


def test_gitops_updater_adds_combined_ai_runtime_and_is_idempotent() -> None:
    image = "qp4534/ev-energy-management-fastapi:new"

    updated = update_manifest(BASE_MANIFEST, image)
    updated_again = update_manifest(updated, image)

    assert updated_again == updated
    assert f"image: {image}" in updated
    assert updated.count("- name: EMBEDDED_AI_ENABLED") == 1
    assert updated.count("- name: REPORT_WORKER_ENABLED") == 1
    assert "key: DEEPSEEK_API_KEY" in updated
    assert 'cpu: "1"' in updated
    assert "memory: 2Gi" in updated
    assert 'cpu: "2"' in updated
    assert "memory: 4Gi" in updated


def test_migration_job_uses_new_image_and_database_secret() -> None:
    manifest = Path("k8s/migration-job.yaml").read_text(encoding="utf-8")
    workflow = Path("../.github/workflows/deploy.yml").read_text(encoding="utf-8")

    assert "REPLACE_WITH_JOB_NAME" in manifest
    assert "REPLACE_WITH_IMAGE_URI" in manifest
    assert 'command: ["alembic", "upgrade", "head"]' in manifest
    assert "key: DATABASE_URL" in manifest
    assert "migration-job.yaml" in workflow
    assert "kubectl wait --for=condition=complete" in workflow
