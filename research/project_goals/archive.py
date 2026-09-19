"""Lossless storage for newly completed poll snapshots; never collector state."""

import hashlib
import json
import os
from pathlib import Path
import shutil
import tarfile
from .common import digest, write_json


def archive_completed(directory, runs, archives, existed_before):
    directory = Path(directory).absolute()
    runs = Path(runs).resolve()
    archives = Path(archives).resolve()
    if (
        directory.is_symlink()
        or directory.resolve() != directory
        or directory in existed_before
    ):
        raise ValueError("only new real invocation directories may be archived")
    adapter = (
        directory.parent == runs / "20260910-contract-feed"
        and directory.name.startswith("poll-")
    )
    shadow = directory.parent == runs and "-shadow-" in directory.name
    if not (adapter or shadow):
        raise ValueError("not a completed operational invocation")
    seal = directory / ("manifest.json" if adapter else "completion.json")
    content = json.loads(seal.read_text())
    declared = content["artifacts" if adapter else "sha256"]
    files = {
        str(p.relative_to(directory)): p for p in directory.rglob("*") if p.is_file()
    }
    if any(p.is_symlink() for p in directory.rglob("*")):
        raise ValueError("symlink in snapshot")
    if set(files) != set(declared) | {seal.name}:
        raise ValueError("completion inventory does not cover snapshot")
    actual = {name: digest(path) for name, path in files.items()}
    if any(actual[name] != sha for name, sha in declared.items()):
        raise ValueError("snapshot completion hash mismatch")
    archives.mkdir(parents=True, exist_ok=True)
    target = archives / (directory.name + ".tar.gz")
    if target.exists():
        raise ValueError("archive already exists; manual recovery required")
    temporary = target.with_suffix(".pending")
    with tarfile.open(temporary, "w:gz", compresslevel=3) as tar:
        for name, path in sorted(files.items()):
            tar.add(path, arcname=name, recursive=False)
    verify_archive(temporary, actual)
    # Verify sources again before removal. Never silently archive a changing snapshot.
    if {name: digest(path) for name, path in files.items()} != actual:
        raise ValueError("snapshot changed during archival")
    with temporary.open("rb") as stream:
        os.fsync(stream.fileno())
    os.replace(temporary, target)
    manifest = dict(
        original_directory=str(directory),
        archive_sha256=digest(target),
        files=actual,
        uncompressed_bytes=sum(p.stat().st_size for p in files.values()),
        compressed_bytes=target.stat().st_size,
        kind="byte_verified_completed_operational_snapshot",
    )
    write_json(target.with_suffix(".json"), manifest)
    with target.with_suffix(".json").open("rb") as stream:
        os.fsync(stream.fileno())
    fd = os.open(archives, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)
    # Read-only sealed modes may prevent non-root cleanup; relax only this verified copy.
    directory.chmod(0o755)
    for path in directory.rglob("*"):
        if path.is_dir():
            path.chmod(0o755)
    shutil.rmtree(directory)
    return manifest


def verify_archive(path, files):
    with tarfile.open(path, "r:gz") as tar:
        members = tar.getmembers()
        if len(members) != len(files) or {m.name for m in members} != set(files):
            raise ValueError("archive member inventory mismatch")
        for member in members:
            if (
                not member.isfile()
                or Path(member.name).is_absolute()
                or ".." in Path(member.name).parts
            ):
                raise ValueError("unsafe archive member")
            h = hashlib.sha256()
            stream = tar.extractfile(member)
            for block in iter(lambda: stream.read(1048576), b""):
                h.update(block)
            if h.hexdigest() != files[member.name]:
                raise ValueError("archive byte verification failed")


def restore(archive, destination):
    archive = Path(archive).resolve()
    destination = Path(destination)
    manifest = json.loads(archive.with_suffix(".json").read_text())
    if digest(archive) != manifest["archive_sha256"]:
        raise ValueError("archive hash mismatch")
    verify_archive(archive, manifest["files"])
    destination.mkdir(parents=True, exist_ok=False)
    with tarfile.open(archive, "r:gz") as tar:
        for member in tar.getmembers():
            path = destination / member.name
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("wb") as stream:
                shutil.copyfileobj(tar.extractfile(member), stream)
    if {name: digest(destination / name) for name in manifest["files"]} != manifest[
        "files"
    ]:
        raise ValueError("restored hash mismatch")
    return manifest


def archive_chunks(directory, runs, archives, existed_before):
    """Content-addressed 1 MiB gzip chunks avoid duplicating unchanged DB pages."""
    import gzip

    directory = Path(directory).absolute()
    runs = Path(runs).resolve()
    archives = Path(archives).resolve()
    if (
        directory.is_symlink()
        or directory.resolve() != directory
        or directory in existed_before
    ):
        raise ValueError("only new real invocation directories may be archived")
    adapter = (
        directory.parent == runs / "20260910-contract-feed"
        and directory.name.startswith("poll-")
    )
    shadow = directory.parent == runs and "-shadow-" in directory.name
    if not (adapter or shadow):
        raise ValueError("not an operational snapshot")
    seal = directory / ("manifest.json" if adapter else "completion.json")
    declared = json.loads(seal.read_text())["artifacts" if adapter else "sha256"]
    paths = {
        str(p.relative_to(directory)): p for p in directory.rglob("*") if p.is_file()
    }
    if any(p.is_symlink() for p in directory.rglob("*")) or set(paths) != set(
        declared
    ) | {seal.name}:
        raise ValueError("unsealed or symlinked snapshot")
    actual = {name: digest(path) for name, path in paths.items()}
    if any(actual[n] != v for n, v in declared.items()):
        raise ValueError("completion hash mismatch")
    chunks_dir = archives / "chunks"
    chunks_dir.mkdir(parents=True, exist_ok=True)
    files = {}
    new_bytes = 0
    for name, path in sorted(paths.items()):
        chunks = []
        with path.open("rb") as stream:
            for raw in iter(lambda: stream.read(1024**2), b""):
                key = hashlib.sha256(raw).hexdigest()
                target = chunks_dir / (key + ".gz")
                if not target.exists():
                    temporary = target.with_suffix(".pending")
                    with temporary.open("wb") as output:
                        output.write(gzip.compress(raw, compresslevel=3, mtime=0))
                        output.flush()
                        os.fsync(output.fileno())
                    os.replace(temporary, target)
                    new_bytes += target.stat().st_size
                if (
                    hashlib.sha256(gzip.decompress(target.read_bytes())).hexdigest()
                    != key
                ):
                    raise ValueError("chunk hash mismatch")
                chunks.append(key)
        files[name] = dict(
            sha256=actual[name],
            size=path.stat().st_size,
            mode=path.stat().st_mode & 0o777,
            chunks=chunks,
        )
    manifest = dict(
        kind="content_addressed_completed_snapshot",
        original_directory=str(directory),
        files=files,
        new_compressed_bytes=new_bytes,
        uncompressed_bytes=sum(p.stat().st_size for p in paths.values()),
    )
    target = archives / (directory.name + ".chunks.json")
    if target.exists():
        raise ValueError("snapshot manifest already exists")
    write_json(target, manifest)
    with target.open("rb") as stream:
        os.fsync(stream.fileno())
    verify_chunks(target)
    if {name: digest(path) for name, path in paths.items()} != actual:
        raise ValueError("source changed during chunk archival")
    for folder in (chunks_dir, archives):
        fd = os.open(folder, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
    directory.chmod(0o755)
    for path in directory.rglob("*"):
        if path.is_dir():
            path.chmod(0o755)
    shutil.rmtree(directory)
    return dict(
        manifest=str(target),
        manifest_sha256=digest(target),
        new_compressed_bytes=new_bytes,
        uncompressed_bytes=manifest["uncompressed_bytes"],
    )


def verify_chunks(manifest_path):
    import gzip

    manifest_path = Path(manifest_path)
    manifest = json.loads(manifest_path.read_text())
    for name, entry in manifest["files"].items():
        if Path(name).is_absolute() or ".." in Path(name).parts:
            raise ValueError("unsafe snapshot member")
        h = hashlib.sha256()
        size = 0
        for key in entry["chunks"]:
            if len(key) != 64 or any(c not in "0123456789abcdef" for c in key):
                raise ValueError("invalid chunk key")
            raw = gzip.decompress(
                (manifest_path.parent / "chunks" / (key + ".gz")).read_bytes()
            )
            if hashlib.sha256(raw).hexdigest() != key:
                raise ValueError("chunk corrupt")
            h.update(raw)
            size += len(raw)
        if h.hexdigest() != entry["sha256"] or size != entry["size"]:
            raise ValueError("reconstructed snapshot mismatch")
    return manifest


def restore_chunks(manifest_path, destination):
    import gzip

    manifest_path = Path(manifest_path)
    manifest = verify_chunks(manifest_path)
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=False)
    for name, entry in manifest["files"].items():
        path = destination / name
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as stream:
            for key in entry["chunks"]:
                stream.write(
                    gzip.decompress(
                        (manifest_path.parent / "chunks" / (key + ".gz")).read_bytes()
                    )
                )
        if digest(path) != entry["sha256"]:
            raise ValueError("restored hash mismatch")
        path.chmod(entry["mode"])
    return manifest
