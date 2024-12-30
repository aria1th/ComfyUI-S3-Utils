import sys
import time
from typing import Tuple
import folder_paths
import boto3
from boto3.s3.transfer import TransferConfig
import comfy.sd
import comfy.utils
import logging
import os

from .s3_manager import s3_client
from .autonode import node_wrapper, validate, get_node_names_mappings

s3_client: boto3.client = None

class ProgressPercentage:
    def __init__(self, file_size):
        self._file_size = file_size
        self._seen_so_far = 0
        self._start_time = time.time()

    def __call__(self, bytes_amount):
        self._seen_so_far += bytes_amount
        elapsed_time = time.time() - self._start_time
        percentage = (self._seen_so_far / self._file_size) * 100
        speed = self._seen_so_far / elapsed_time if elapsed_time > 0 else 0
        sys.stdout.write(
            f"\rDownloading... {self._seen_so_far}/{self._file_size} bytes "
            f"({percentage:.2f}%) | Speed: {speed:.2f} bytes/s | Elapsed: {elapsed_time:.2f}s"
        )
        sys.stdout.flush()

def get_lora_from_s3(bucket_name: str, object_key: str, save_path: str) -> bool:
    try:
        logging.info(f"Fetching LoRA from S3: {bucket_name}/{object_key} to {save_path}")

        response = s3_client.head_object(Bucket=bucket_name, Key=object_key)
        file_size = response['ContentLength']
        config = TransferConfig(multipart_threshold=1024 * 25, max_concurrency=10, multipart_chunksize=1024 * 25)
        start_time = time.time()
        s3_client.download_file(
            bucket_name, object_key, save_path,
            Config=config,
            Callback=ProgressPercentage(file_size)
        )
        elapsed_time = time.time() - start_time
        logging.info(f"Successfully saved LoRA file to {save_path} in {elapsed_time:.2f} seconds.")
        return True
    except Exception as e:
        logging.error(f"Error fetching LoRA from S3: {e}")
        return False

def get_full_path_simulate(folder_name: str, filename: str) -> Tuple[str, bool]:
    """
    Simulates the full path of the file if it exists.
    Returns the full path and a boolean indicating if the file exists.
    """
    # validate folder_name
    folder_name = ensure_path_safety(folder_name)
    filename = ensure_path_safety(filename)
    folder_name = folder_paths.map_legacy(folder_name)
    if folder_name not in folder_paths.folder_names_and_paths:
        raise FileNotFoundError(f"Folder '{folder_name}' not found in folder_names_and_paths.")
    folders = folder_paths.folder_names_and_paths[folder_name]
    filename = os.path.relpath(os.path.join("/", filename), "/")
    for x in folders[0]:
        full_path = os.path.join(x, filename)
        if os.path.isfile(full_path):
            return full_path, True
        elif os.path.islink(full_path):
            logging.warning("WARNING path {} exists but doesn't link anywhere, skipping.".format(full_path))
    # fallback to first folder
    full_path = os.path.join(folders[0][0], filename)
    if not os.path.isfile(full_path):
        return full_path, False
    raise RuntimeError(f"This condition should never be reached, {full_path} exists but is not a file.")

def get_full_path_or_raise(folder_name: str, filename: str, bucket_name: str, object_key: str) -> str: 
    """
    Fetches the full path of the file if it exists, otherwise fetches the file from S3.
    """
    full_path, exists = get_full_path_simulate(folder_name, filename)
    if not exists:
        if not get_lora_from_s3(bucket_name, object_key, full_path):
            raise FileNotFoundError(f"Failed to fetch LoRA from S3: {bucket_name}/{object_key}")
        return full_path
    return full_path

def ensure_path_safety(path: str) -> str:
    # Ensure path is safe, .. and ~ are not allowed
    if ".." in path or "~" in path:
        raise ValueError(f"Path '{path}' is not safe.")
    return path

classes = []
wrapper = node_wrapper(classes)

@wrapper
class LoraLoaderFallbackS3:
    def __init__(self):
        self.loaded_lora = None

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": { 
                "model": ("MODEL", {"tooltip": "The diffusion model the LoRA will be applied to."}),
                "clip": ("CLIP", {"tooltip": "The CLIP model the LoRA will be applied to."}),
                "lora_name": ("STRING", {"tooltip": "The name of the LoRA to load."}),
                "strength_model": ("FLOAT", {"default": 1.0, "min": -100.0, "max": 100.0, "step": 0.01, "tooltip": "How strongly to modify the diffusion model. This value can be negative."}),
                "strength_clip": ("FLOAT", {"default": 1.0, "min": -100.0, "max": 100.0, "step": 0.01, "tooltip": "How strongly to modify the CLIP model. This value can be negative."}),
                "bucket_name": ("STRING", {"default": "", "tooltip": "The name of the bucket where the LoRA is stored."}),
                "object_key": ("STRING", {"default": "", "tooltip": "The object key for the LoRA."}),
            }
        }
    
    RETURN_TYPES = ("MODEL", "CLIP")
    OUTPUT_TOOLTIPS = ("The modified diffusion model.", "The modified CLIP model.")
    FUNCTION = "load_lora"
    CATEGORY = "loaders"
    DESCRIPTION = "LoRAs are used to modify diffusion and CLIP models, altering the way in which latents are denoised such as applying styles. Multiple LoRA nodes can be linked together. Here, we use S3 to fetch the LoRA file."
    custom_name = "Load LoRA (Fallback S3)"

    def load_lora(self, model, clip, lora_name, strength_model, strength_clip, bucket_name, object_key):
        if strength_model == 0 and strength_clip == 0:
            return (model, clip)

        lora_path = get_full_path_or_raise("loras", lora_name, bucket_name, object_key)
        lora = None
        if self.loaded_lora is not None:
            if self.loaded_lora[0] == lora_path:
                lora = self.loaded_lora[1]
            else:
                temp = self.loaded_lora
                self.loaded_lora = None
                del temp

        if lora is None:
            lora = comfy.utils.load_torch_file(lora_path, safe_load=True)
            self.loaded_lora = (lora_path, lora)

        model_lora, clip_lora = comfy.sd.load_lora_for_models(model, clip, lora, strength_model, strength_clip)
        return (model_lora, clip_lora)

@wrapper
class LoraLoaderModelOnlyFallbackS3(LoraLoaderFallbackS3):
    @classmethod
    def INPUT_TYPES(s):
        return {"required": { "model": ("MODEL",),
                                "lora_name": (folder_paths.get_filename_list("loras"), ),
                                "strength_model": ("FLOAT", {"default": 1.0, "min": -100.0, "max": 100.0, "step": 0.01}),
                                "bucket_name": ("STRING", {"default": ""}),
                                "object_key": ("STRING", {"default": ""}),
                            }}
    RETURN_TYPES = ("MODEL",)
    FUNCTION = "load_lora_model_only"
    custom_name = "Load LoRA Model Only (Fallback S3)"
    def load_lora_model_only(self, model, lora_name, strength_model, bucket_name, object_key):
        return (self.load_lora(model, None, lora_name, strength_model, 0, bucket_name, object_key)[0],)

validate(classes)
CLASS_MAPPINGS, CLASS_NAMES = get_node_names_mappings(classes)
