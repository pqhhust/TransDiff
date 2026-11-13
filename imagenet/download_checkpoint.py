"""Download only one pinned ViT weights format, outside the shared read-only area."""
import argparse
import hashlib
import json
from pathlib import Path
from huggingface_hub import snapshot_download

REPO = 'google/vit-large-patch16-224'
REVISION = '9e2727f4250d3973839eecfa5c4b42e41b709a50'


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', default=str(Path.home()/'models'/REPO))
    a = p.parse_args()
    target = Path(a.output).expanduser()
    snapshot_download(REPO, revision=REVISION, local_dir=target,
                      allow_patterns=['pytorch_model.bin', 'config.json', 'preprocessor_config.json', 'README.md'])
    with (target/'pytorch_model.bin').open('rb') as f:
        digest = hashlib.file_digest(f, 'sha256').hexdigest()
    (target/'provenance.json').write_text(json.dumps(dict(repo=REPO, revision=REVISION, sha256=digest), indent=2)+'\n')
    print(target)


if __name__ == '__main__':
    main()
