from pathlib import Path

from scripts.update_gitops_fastapi_deployment import update_manifest


BASE_MANIFEST = """\
apiVersion: apps/v1
kind: Deployment
spec:
  replicas: 1
  strategy:
    type: RollingUpdate
    rollingUpdate:
      maxUnavailable: 0
      maxSurge: 1
  selector:
    matchLabels:
      app: ev-ai-inference-api
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
    assert "maxUnavailable: 1" in updated
    assert "maxSurge: 0" in updated
    assert "maxUnavailable: 0" not in updated


def test_migration_job_uses_new_image_and_database_secret() -> None:
    manifest = Path("k8s/migration-job.yaml").read_text(encoding="utf-8")
    workflow = Path("../.github/workflows/deploy.yml").read_text(encoding="utf-8")

    assert "REPLACE_WITH_JOB_NAME" in manifest
    assert "REPLACE_WITH_IMAGE_URI" in manifest
    assert 'command: ["alembic", "upgrade", "head"]' in manifest
    assert "key: DATABASE_URL" in manifest
    assert "ttlSecondsAfterFinished: 3600" in manifest
    assert "migration-job.yaml" in workflow
    assert ".status.succeeded" in workflow
    assert ".status.failed" in workflow
    assert "kubectl logs job/" in workflow


def test_runtime_image_uses_a_writable_huggingface_cache() -> None:
    dockerfile = Path("Dockerfile").read_text(encoding="utf-8")

    assert "HOME=/app" in dockerfile
    assert "HF_HOME=/app/.cache/huggingface" in dockerfile
    assert "mkdir -p /app/.cache/huggingface" in dockerfile
    assert "chown -R app:app /app/.cache" in dockerfile


def test_runtime_image_forces_cpu_only_torch() -> None:
    dockerfile = Path("Dockerfile").read_text(encoding="utf-8")

    assert "ARG TORCH_VERSION=2.13.0" in dockerfile
    assert "https://download.pytorch.org/whl/cpu" in dockerfile
    assert '"torch==${TORCH_VERSION}+cpu"' in dockerfile
    assert "torch.version.cuda is None" in dockerfile
