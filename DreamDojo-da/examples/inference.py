

from pathlib import Path
from typing import Annotated

import pydantic
import tyro
from cosmos_oss.init import cleanup_environment, init_environment, init_output_dir

from cosmos_predict2.config import (
    InferenceArguments,
    InferenceOverrides,
    SetupArguments,
    handle_tyro_exception,
    is_rank0,
)

class Args(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(extra="forbid", frozen=True)

    input_files: Annotated[list[Path], tyro.conf.arg(aliases=("-i",))]
    setup: SetupArguments
    overrides: InferenceOverrides

def main(
    args: Args,
):
    inference_samples = InferenceArguments.from_files(args.input_files, overrides=args.overrides)
    init_output_dir(args.setup.output_dir, profile=args.setup.profile)

    from cosmos_predict2.inference import Inference

    inference = Inference(args.setup)
    inference.generate(inference_samples, output_dir=args.setup.output_dir)

if __name__ == "__main__":
    init_environment()

    try:
        args = tyro.cli(
            Args,
            description=__doc__,
            console_outputs=is_rank0(),
            config=(tyro.conf.OmitArgPrefixes,),
        )
    except Exception as e:
        handle_tyro_exception(e)

    main(args)

    cleanup_environment()
