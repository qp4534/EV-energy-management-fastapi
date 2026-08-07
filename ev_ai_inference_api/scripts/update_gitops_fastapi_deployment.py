from __future__ import annotations

import argparse
import re
from pathlib import Path


AI_ENVIRONMENT = """\
            - name: EMBEDDED_AI_ENABLED
              value: "true"
            - name: REPORT_JOBS_ENABLED
              value: "true"
            - name: REPORT_WORKER_ENABLED
              value: "true"
            - name: DEEPSEEK_API_KEY
              valueFrom:
                secretKeyRef:
                  name: fastapi-secret
                  key: DEEPSEEK_API_KEY
                  optional: true
"""

AI_RESOURCES = """\
          resources:
            requests:
              cpu: "1"
              memory: 2Gi
            limits:
              cpu: "2"
              memory: 4Gi
"""

SINGLE_NODE_STRATEGY = """\
  strategy:
    type: RollingUpdate
    rollingUpdate:
      maxUnavailable: 1
      maxSurge: 0
"""


def update_manifest(content: str, image: str) -> str:
    content, image_count = re.subn(
        r"(?m)^(\s*image:\s*)\S+\s*$",
        rf"\g<1>{image}",
        content,
        count=1,
    )
    if image_count != 1:
        raise ValueError("FastAPI image line was not found exactly once")

    strategy_pattern = re.compile(
        r"(?m)^  strategy:\n(?: {4,}.*\n)+(?=  selector:)"
    )
    if strategy_pattern.search(content):
        content = strategy_pattern.sub(SINGLE_NODE_STRATEGY, content, count=1)
    else:
        selector_marker = "  selector:"
        if content.count(selector_marker) != 1:
            raise ValueError("FastAPI deployment selector was not found exactly once")
        content = content.replace(
            selector_marker,
            SINGLE_NODE_STRATEGY + selector_marker,
            1,
        )

    if "- name: EMBEDDED_AI_ENABLED" not in content:
        marker = "          resources:\n"
        if content.count(marker) != 1:
            raise ValueError("FastAPI resources block was not found exactly once")
        content = content.replace(marker, AI_ENVIRONMENT + marker, 1)

    resources_pattern = re.compile(
        r"          resources:\n"
        r"            requests:\n"
        r"              cpu: [^\n]+\n"
        r"              memory: [^\n]+\n"
        r"            limits:\n"
        r"              cpu: [^\n]+\n"
        r"              memory: [^\n]+\n"
    )
    content, resource_count = resources_pattern.subn(
        AI_RESOURCES,
        content,
        count=1,
    )
    if resource_count != 1:
        raise ValueError("FastAPI resources values were not found exactly once")
    return content


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=Path, required=True)
    parser.add_argument("--image", required=True)
    args = parser.parse_args()

    original = args.path.read_text(encoding="utf-8")
    updated = update_manifest(original, args.image)
    args.path.write_text(updated, encoding="utf-8", newline="\n")


if __name__ == "__main__":
    main()
