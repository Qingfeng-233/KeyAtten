from __future__ import annotations

import argparse
import tarfile
from pathlib import Path

import paramiko


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Upload and run hidden-head training on remote host via SSH."
    )
    parser.add_argument("--host", required=True)
    parser.add_argument("--port", type=int, default=22)
    parser.add_argument("--username", required=True)
    parser.add_argument("--password", required=True)
    parser.add_argument("--remote-dir", default="/root/hidden_head_run")
    parser.add_argument("--bundle-name", default="hidden_head_remote_bundle.tar.gz")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--train-limit", type=int, default=300)
    parser.add_argument("--dev-limit", type=int, default=250)
    return parser.parse_args()


def create_bundle(bundle_path: Path) -> None:
    root = Path(__file__).resolve().parents[2]
    bench = root / "benchmark"
    members = [
        root / "train" / "train_hidden_state_head.py",
        bench / "eval" / "run_hidden_head_benchmark.py",
        bench / "keyword_bench",
        root / "测试沙箱" / "external" / "CSL" / "benchmark" / "kg",
        root / "测试沙箱" / "models" / "thenlper__gte-small-zh",
    ]

    with tarfile.open(bundle_path, "w:gz") as tar:
        for src in members:
            if not src.exists():
                raise FileNotFoundError(f"Missing required path: {src}")
            arcname = src.relative_to(root).as_posix()
            tar.add(src, arcname=arcname)


def run_remote(
    ssh: paramiko.SSHClient, command: str, timeout: int = 1200
) -> tuple[int, str, str]:
    stdin, stdout, stderr = ssh.exec_command(command, timeout=timeout)
    code = stdout.channel.recv_exit_status()
    return (
        code,
        stdout.read().decode("utf-8", "ignore"),
        stderr.read().decode("utf-8", "ignore"),
    )


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parents[2]
    bundle_path = root / args.bundle_name
    create_bundle(bundle_path)
    print(f"Bundle created: {bundle_path} ({bundle_path.stat().st_size} bytes)")

    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect(
        hostname=args.host,
        port=args.port,
        username=args.username,
        password=args.password,
        timeout=20,
        auth_timeout=20,
        banner_timeout=20,
    )
    sftp = ssh.open_sftp()
    remote_bundle = f"{args.remote_dir}/{args.bundle_name}"
    run_remote(ssh, f"mkdir -p {args.remote_dir}")
    sftp.put(str(bundle_path), remote_bundle)
    sftp.close()
    print(f"Uploaded: {remote_bundle}")

    commands = [
        f"mkdir -p {args.remote_dir}/bundle",
        f"tar -xzf {remote_bundle} -C {args.remote_dir}/bundle",
        (
            "apt-get update && "
            "DEBIAN_FRONTEND=noninteractive apt-get install -y python3-pip python3-venv && "
            "python3 -m pip install -U pip && "
            "python3 -m pip install torch transformers jieba numpy"
        ),
        (
            f"cd {args.remote_dir}/bundle && "
            f"python3 train/train_hidden_state_head.py "
            f"--root-dir 测试沙箱 "
            f"--output-dir outputs_hidden_head_remote_formal "
            f"--model 测试沙箱/models/thenlper__gte-small-zh "
            f"--train-dataset csl_train_sample --dev-dataset csl_dev "
            f"--train-limit {args.train_limit} --dev-limit {args.dev_limit} "
            f"--epochs {args.epochs} --batch-size {args.batch_size} "
            f"--learning-rate {args.learning_rate} --device cuda"
        ),
        (
            f"cd {args.remote_dir}/bundle && "
            f"python3 benchmark/run_hidden_head_benchmark.py "
            f"--root-dir 测试沙箱 "
            f"--output-dir outputs_hidden_head_remote_eval "
            f"--checkpoint 测试沙箱/Outputs/outputs_hidden_head_remote_formal/best_hidden_head.pt "
            f"--model 测试沙箱/models/thenlper__gte-small-zh "
            f"--datasets csl_dev csl_test --top-k 10 --device cuda --batch-size 8"
        ),
    ]

    for index, command in enumerate(commands, start=1):
        print(f"\n[{index}/{len(commands)}] {command}")
        code, out, err = run_remote(ssh, command, timeout=7200)
        print(out)
        if err.strip():
            print(err)
        if code != 0:
            raise RuntimeError(
                f"Remote command failed with exit code {code}: {command}"
            )

    ssh.close()
    print("Remote training + eval completed.")


if __name__ == "__main__":
    main()
