

from pathlib import Path

import pydantic
import subprocess
import time
import torch
import tyro
from cosmos_oss.init import cleanup_environment, init_environment, init_output_dir

from cosmos_predict2.action_conditioned_config import (
    ActionConditionedInferenceArguments,
    ActionConditionedSetupArguments,
)
from cosmos_predict2.config import (
    handle_tyro_exception,
    is_rank0,
)

class Args(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(extra="forbid", frozen=True)

    setup: ActionConditionedSetupArguments

def convert_checkpoint(checkpoint_iter_dir: Path):

    subprocess.run(
        ["sudo", "chmod", "-R", "777", str(checkpoint_iter_dir)],
        check=True,
        capture_output=True
    )

    result = subprocess.run(
        ["python", "./scripts/convert_distcp_to_pt.py", str(checkpoint_iter_dir / "model"), str(checkpoint_iter_dir)],
        check=True,
        capture_output=True
    )
    print(result.stdout)

def main(args: Args) -> None:
    print("=== main() entered ===", flush=True)
    inference_args = ActionConditionedInferenceArguments
    init_output_dir(args.setup.output_dir, profile=args.setup.profile)
    print("=== init_output_dir done ===", flush=True)

    from cosmos_predict2.action_conditioned import inference

    last_checkpoint_file = Path(args.setup.checkpoints_dir) / "latest_checkpoint.txt"
    f = open(last_checkpoint_file, "r")
    last_checkpoint = f.read().strip()
    f.close()

    if args.setup.infinite:
        last_iter = int(last_checkpoint.split("_")[-1])
        interval = args.setup.checkpoint_interval

        while True:
            try:
                for iter in range(interval, last_iter + interval * 100, interval):
                    infer_results_dir = Path(args.setup.save_dir) / f"iter_{iter:09d}" / "all_summary.json"
                    if infer_results_dir.exists():
                        continue
                    legacy_infer_results_dir = Path(args.setup.save_dir) / f"iter_{iter:09d}" / "metrics.json"
                    if legacy_infer_results_dir.exists():
                        continue
                    checkpoint_iter_dir = Path(args.setup.checkpoints_dir) / f"iter_{iter:09d}"
                    if not checkpoint_iter_dir.exists():
                        continue
                    checkpoint_path = checkpoint_iter_dir / "model_ema_bf16.pt"
                    if not checkpoint_path.exists():
                        convert_checkpoint(checkpoint_iter_dir)
                    inference(args.setup, inference_args, checkpoint_path)
            except Exception as e:
                pass
            time.sleep(60)
    else:
        checkpoint_iter_dir = Path(args.setup.checkpoints_dir) / last_checkpoint
        checkpoint_path = checkpoint_iter_dir / "model_ema_bf16.pt"
        if not checkpoint_path.exists():

            if is_rank0():
                convert_checkpoint(checkpoint_iter_dir)
            if torch.distributed.is_initialized():
                torch.distributed.barrier()
        inference(args.setup, inference_args, checkpoint_path)

if __name__ == "__main__":
    print("=== Script started ===", flush=True)
    init_environment()
    print("=== init_environment done ===", flush=True)

    try:
        args = tyro.cli(
            Args,
            description=__doc__,
            console_outputs=is_rank0(),
            config=(tyro.conf.OmitArgPrefixes,),
        )
    except Exception as e:
        handle_tyro_exception(e)
    print("=== args parsed, calling main ===", flush=True)

    main(args)

    cleanup_environment()
