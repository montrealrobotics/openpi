from collections.abc import Sequence
import logging
import pathlib
import time
from typing import Any, TypeAlias

import flax
import flax.traverse_util
import jax
import jax.numpy as jnp
import numpy as np
from openpi_client import base_policy as _base_policy
from PIL import Image
from io import BytesIO
import requests

import torch
from typing_extensions import override


from transformers import AutoProcessor, AutoModelForImageTextToText

BasePolicy: TypeAlias = _base_policy.BasePolicy

def convert_image(img):
    image_converted = img.astype(np.uint8)

    if image_converted.ndim == 2:
        image_converted = np.stack((image_converted,)*3, axis=-1)

    if image_converted.shape[-1] == 4:
        image_converted = image_converted[..., :3]
    return image_converted

class Policy(BasePolicy):
    def __init__(
        self,
        checkpoint: str = "allenai/MolmoAct-7B-D-0812",
    ):
        """Initialize the Policy.

        Args:
            model: The model to use for action sampling.
            rng: Random number generator key for JAX models. Ignored for PyTorch models.
            transforms: Input data transformations to apply before inference.
            output_transforms: Output data transformations to apply after inference.
            sample_kwargs: Additional keyword arguments to pass to model.sample_actions.
            metadata: Additional metadata to store with the policy.
            pytorch_device: Device to use for PyTorch models (e.g., "cpu", "cuda:0").
                          Only relevant when is_pytorch=True.
            is_pytorch: Whether the model is a PyTorch model. If False, assumes JAX model.
        """
        self.ckpt = checkpoint
        device = "cuda"
        self.processor = AutoProcessor.from_pretrained(
            self.ckpt,
            trust_remote_code=True,
            torch_dtype="bfloat16",
            device_map={ "": device },
            padding_side="left",
        )

        # load the model
        self.model = AutoModelForImageTextToText.from_pretrained(
            self.ckpt,
            trust_remote_code=True,
            torch_dtype="bfloat16",
            device_map=None,
        ).to(device)


    @override
    def infer(self, obs: dict, *, noise: np.ndarray | None = None) -> dict:  # type: ignore[misc]
        inputs = obs
        print("inference started")

        gripper_pos = np.asarray(inputs["observation/gripper_position"])
        if gripper_pos.ndim == 0:
            # Ensure gripper position is a 1D array, not a scalar, so we can concatenate with joint positions
            gripper_pos = gripper_pos[np.newaxis]

        state = np.concatenate([inputs["observation/joint_position"], gripper_pos])

        im1 = inputs["observation/exterior_image_1_left"]
        wrist_image = inputs["observation/wrist_image_left"]
        instruction = inputs["prompt"]
        
        img1 = convert_image(im1)
        img2 = convert_image(wrist_image)

        imgs = [img1, img2]
        #url1 = "https://huggingface.co/allenai/MolmoAct-7B-D-0812/resolve/main/example_1.png"
        #url2 = "https://huggingface.co/allenai/MolmoAct-7B-D-0812/resolve/main/example_2.png"
        #r1 = requests.get(url1, headers={"User-Agent": "python-requests"}, timeout=30)
        #r1.raise_for_status()
        #r2 = requests.get(url2, headers={"User-Agent": "python-requests"}, timeout=30)
        #r2.raise_for_status()
        #img1 = Image.open(BytesIO(r1.content)).convert("RGB")
        #img2 = Image.open(BytesIO(r2.content)).convert("RGB")
        #imgs = [img1, img2]

        prompt = (
            f"The task is {instruction}. "
            "What is the action that the robot should take. "
            f"To figure out the action that the robot should take to {instruction}, "
            "let's think through it step by step. "
            "First, what is the depth map for the first image? "
            "Second, what is the trajectory of the end effector in the first image? "
            "Based on the depth map of the first image and the trajectory of the end effector in the first image, "
            "along with other images from different camera views as additional information, "
            "what is the action that the robot should take?"
        )
        prompt1 = (
            f"The task is {instruction}. "
            "What is the trajectory that the end effector should take? "
        )
        text = self.processor.apply_chat_template(
            [
                {
                    "role": "user",
                    "content": [dict(type="text", text=prompt)]
                }
            ], 
            tokenize=False, 
            add_generation_prompt=True,
        )
        inputs = self.processor(
            images=[imgs],
            text=text,
            padding=True,
            return_tensors="pt",
        )

        # move inputs to the correct device
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        # generate output
        with torch.inference_mode():
            with torch.autocast("cuda", enabled=True, dtype=torch.bfloat16):
                generated_ids = self.model.generate(**inputs, max_new_tokens=256)

        # only get generated tokens; decode them to text
        generated_tokens = generated_ids[:, inputs['input_ids'].size(1):]
        generated_text = self.processor.batch_decode(generated_tokens, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]

        # print the generated text
        print(f"generated text: {generated_text}")

        # >>>  The depth map of the first image is ... The trajectory of the end effector in the first image is ...
        #      Based on these information, along with other images from different camera views as additional information,
        #      the action that the robot should take is ...

        # parse out all depth perception tokens
        depth = self.model.parse_depth(generated_text)
        print(f"generated depth perception tokens: {depth}")

        # >>>  [ "<DEPTH_START><DEPTH_1><DEPTH_2>...<DEPTH_END>" ]

        # parse out all visual reasoning traces
        trace = self.model.parse_trace(generated_text)
        print(f"generated visual reasoning trace: {trace}")

        # >>>  [ [[242, 115], [140, 77], [94, 58], [140, 44], [153, 26]]] ]

        # parse out all actions, unnormalizing with key of "molmoact"
        action = self.model.parse_action(generated_text, unnorm_key="molmoact")
        start_time = time.monotonic()
        outputs = {
            "state": state,
            "action": np.array(action),
            "trace": trace,
        }
        model_time = time.monotonic() - start_time

        outputs["policy_timing"] = {
            "infer_ms": model_time * 1000,
        }
        return outputs

    @property
    def metadata(self) -> dict[str, Any]:
        return self._metadata


class PolicyRecorder(_base_policy.BasePolicy):
    """Records the policy's behavior to disk."""

    def __init__(self, policy: _base_policy.BasePolicy, record_dir: str):
        self._policy = policy

        logging.info(f"Dumping policy records to: {record_dir}")
        self._record_dir = pathlib.Path(record_dir)
        self._record_dir.mkdir(parents=True, exist_ok=True)
        self._record_step = 0

    @override
    def infer(self, obs: dict) -> dict:  # type: ignore[misc]
        results = self._policy.infer(obs)

        data = {"inputs": obs, "outputs": results}
        data = flax.traverse_util.flatten_dict(data, sep="/")

        output_path = self._record_dir / f"step_{self._record_step}"
        self._record_step += 1

        np.save(output_path, np.asarray(data))
        return results
