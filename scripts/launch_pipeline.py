"""自包含流水线启动脚本。

参数全部来自 configs/launch.yaml，CLI 只保留 --config / --override。
设计目标：在 4/8 张 5090 上把「抽特征 → 比 backbone → 微调 → 评估」串起来跑，
同时让卡数、候选、实验配置都可以只改 YAML 不调脚本。

用法
----
    python scripts/launch_pipeline.py --config configs/launch.yaml

阶段说明
--------
- prepare_data:   单进程，CPU。
- extract_features: 按候选循环分配到各 GPU，每张卡一个进程并行抽特征。
- linear_probe:   单进程，在缓存特征上跑，很快。
- train:          用 torchrun 启动 DDP，占用 gpu_ids 里全部卡。
- evaluate:       单进程，读取 train 保存的 best.pt。
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import Config, add_config_args, config_from_args  # noqa: E402


def run_cmd(cmd: list[str], log_path: Path, env: dict[str, str] | None = None) -> int:
    """运行命令，stdout/stderr 同时打到终端和日志文件。"""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    merged_env = {**os.environ, **env} if env else None
    with log_path.open("w", encoding="utf-8") as f:
        f.write(f"$ {' '.join(cmd)}\n")
        f.write(f"env: {env or {}}\n")
        f.write("=" * 70 + "\n")
        f.flush()
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            env=merged_env,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            print(line, end="")
            f.write(line)
            f.flush()
        proc.wait()
    return proc.returncode


def stage_prepare_data(cfg: Config, log_dir: Path) -> None:
    """阶段 0：数据划分。"""
    print("\n" + "=" * 70)
    print("阶段 0/4：prepare_data")
    print("=" * 70)
    cmd = [
        sys.executable,
        "scripts/prepare_data.py",
        "--config",
        cfg["prepare_data.config"],
    ]
    rc = run_cmd(cmd, log_dir / "00_prepare_data.log")
    if rc != 0:
        raise RuntimeError(f"prepare_data.py 失败，返回码 {rc}")


def _extract_one(candidate: str, gpu_id: int, cfg: Config, log_dir: Path) -> tuple[str, int]:
    """在一个 GPU 上抽一个候选的特征。"""
    cmd = [
        sys.executable,
        "scripts/extract_features.py",
    ]
    for c in cfg["extract_features.configs"]:
        cmd.extend(["--config", c])
    cmd.extend(["--only", candidate])
    log_path = log_dir / f"01_extract_features_{candidate}_gpu{gpu_id}.log"
    rc = run_cmd(cmd, log_path, env={"CUDA_VISIBLE_DEVICES": str(gpu_id)})
    return candidate, rc


def stage_extract_features(cfg: Config, log_dir: Path) -> None:
    """阶段 1：并行抽特征。

    候选按 gpu_ids 循环分配。例如 8 卡 6 候选，则 0~5 号卡各跑一个；
    4 卡 6 候选，则先并行跑 4 个，剩下的 2 个等前面释放后继续。
    """
    print("\n" + "=" * 70)
    print("阶段 1/4：extract_features（并行）")
    print("=" * 70)

    candidates: list[str] = cfg["extract_features.candidates"]
    gpu_ids: list[int] = cfg["gpu_ids"]
    continue_on_error: bool = cfg.get_path("extract_features.continue_on_error", True)

    if not candidates:
        print("extract_features.candidates 为空，跳过")
        return

    # 用线程池控制并发数 = GPU 数，每个线程绑定一张卡跑一个候选。
    results: list[tuple[str, int]] = []
    with ThreadPoolExecutor(max_workers=len(gpu_ids)) as executor:
        futures = {
            executor.submit(
                _extract_one, cand, gpu_ids[i % len(gpu_ids)], cfg, log_dir
            ): cand
            for i, cand in enumerate(candidates)
        }
        for future in as_completed(futures):
            cand, rc = future.result()
            results.append((cand, rc))
            if rc != 0:
                print(f"[warn] {cand} 特征抽取失败，返回码 {rc}")
                if not continue_on_error:
                    raise RuntimeError(f"extract_features {cand} 失败，已停止")

    failed = [c for c, rc in results if rc != 0]
    if failed:
        print(f"[warn] 以下候选抽取失败：{failed}")
    else:
        print("所有候选特征抽取完成")


def stage_linear_probe(cfg: Config, log_dir: Path) -> None:
    """阶段 2：backbone 大比拼。"""
    print("\n" + "=" * 70)
    print("阶段 2/4：linear_probe")
    print("=" * 70)
    cmd = [
        sys.executable,
        "scripts/linear_probe.py",
    ]
    for c in cfg["linear_probe.configs"]:
        cmd.extend(["--config", c])
    rc = run_cmd(cmd, log_dir / "02_linear_probe.log")
    if rc != 0:
        raise RuntimeError(f"linear_probe.py 失败，返回码 {rc}")


def stage_train(cfg: Config, log_dir: Path) -> None:
    """阶段 3：多卡微调。"""
    print("\n" + "=" * 70)
    print("阶段 3/4：train（DDP）")
    print("=" * 70)

    gpu_ids: list[int] = cfg["gpu_ids"]
    master_port: int = cfg["master_port"]
    nproc = len(gpu_ids)

    cmd = [
        "torchrun",
        f"--nproc_per_node={nproc}",
        f"--master_port={master_port}",
        "scripts/train.py",
    ]
    for c in cfg["train.configs"]:
        cmd.extend(["--config", c])

    overrides = cfg.get_path("train.overrides", {}) or {}
    for key, value in overrides.items():
        cmd.extend(["--override", f"{key}={value}"])

    env = {"CUDA_VISIBLE_DEVICES": ",".join(str(g) for g in gpu_ids)}
    rc = run_cmd(cmd, log_dir / "03_train.log", env=env)
    if rc != 0:
        raise RuntimeError(f"train.py 失败，返回码 {rc}")


def stage_evaluate(cfg: Config, log_dir: Path) -> None:
    """阶段 4：评估 best.pt。"""
    print("\n" + "=" * 70)
    print("阶段 4/4：evaluate")
    print("=" * 70)

    checkpoint_template: str = cfg["evaluate.checkpoint"]
    train_checkpoint_dir = cfg.get_path("train.overrides.train.checkpoint_dir", None)
    if train_checkpoint_dir is None:
        # 从训练配置里找 checkpoint_dir
        train_cfg_path = cfg["train.configs"][-1]
        import yaml
        with Path(train_cfg_path).open("r", encoding="utf-8") as f:
            train_cfg = yaml.safe_load(f)
        train_checkpoint_dir = (
            train_cfg.get("train", {}).get("checkpoint_dir") or "checkpoints"
        )

    checkpoint = checkpoint_template.format(train={"checkpoint_dir": train_checkpoint_dir})

    cmd = [
        sys.executable,
        "scripts/evaluate.py",
        "--config",
        cfg["evaluate.config"],
        "--override",
        f"eval.checkpoint={checkpoint}",
    ]
    rc = run_cmd(cmd, log_dir / "04_evaluate.log")
    if rc != 0:
        raise RuntimeError(f"evaluate.py 失败，返回码 {rc}")


def main() -> None:
    parser = argparse.ArgumentParser(description="自包含多 GPU 流水线启动")
    parser = add_config_args(parser)
    args = parser.parse_args()

    cfg: Config = config_from_args(args)
    log_dir = Path(cfg.get_path("log_dir", "outputs/launch_logs"))

    start_time = time.time()
    try:
        if cfg.get_path("stages.prepare_data", False):
            stage_prepare_data(cfg, log_dir)

        if cfg.get_path("stages.extract_features", False):
            stage_extract_features(cfg, log_dir)

        if cfg.get_path("stages.linear_probe", False):
            stage_linear_probe(cfg, log_dir)

        if cfg.get_path("stages.train", False):
            stage_train(cfg, log_dir)

        if cfg.get_path("stages.evaluate", False):
            stage_evaluate(cfg, log_dir)

        elapsed = time.time() - start_time
        print("\n" + "=" * 70)
        print(f"流水线全部完成，总耗时 {elapsed / 60:.1f} 分钟")
        print(f"日志目录：{log_dir.resolve()}")
        print("=" * 70)

    except Exception as exc:
        elapsed = time.time() - start_time
        print("\n" + "=" * 70)
        print(f"流水线中断：{exc}")
        print(f"已运行 {elapsed / 60:.1f} 分钟，日志目录：{log_dir.resolve()}")
        print("=" * 70)
        raise


if __name__ == "__main__":
    main()
