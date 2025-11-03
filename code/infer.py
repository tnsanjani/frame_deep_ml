import os
import torch
import argparse
import copy
from diffusers.utils import load_image, export_to_video
from diffusers import UNetSpatioTemporalConditionModel
from custom_diffusers.pipelines.pipeline_frame_interpolation_with_noise_injection import FrameInterpolationWithNoiseInjectionPipeline
#from custom_diffusers.pipelines.evs_pipeline_frame_interpolation_with_noise_injection_color import EVSFrameInterpolationWithNoiseInjectionPipeline

from custom_diffusers.pipelines.evs_pipeline_frame_interpolation_lefmodel import EVSFrameInterpolationWithNoiseInjectionPipeline
from custom_diffusers.schedulers.scheduling_euler_discrete import EulerDiscreteScheduler
from attn_ctrl.attention_control import (AttentionStore, 
                                         register_temporal_self_attention_control, 
                                         register_temporal_self_attention_flip_control)

from torch.utils.data import DataLoader
from einops import rearrange

import numpy as np
import cv2

from PIL import Image
import torch
from data import StereoEventDataset 
from torch.utils.data import DataLoader
import torch
import torchvision.transforms as T
import cv2
import numpy as np
from accelerate import Accelerator


TARGET_HEIGHT = 576
TARGET_WIDTH = 1024

class StereoEventTestDataset(StereoEventDataset):
    def __init__(self, video_data_dir, frame_height=375, frame_width=375):
        # 375x375 is passed to super() for its cropping logic
        super().__init__(video_data_dir, frame_height, frame_width)
        
        # --- FIX: Make dataset robust ---
        # Check if videos were found, then keep only the first one
        if self.video_names:
            self.video_names = [self.video_names[0]]
            self.length = 1
        else:
            self.length = 0
            print("Warning: No videos found in the specified directory.")
        # --- END FIX ---
            
        self.resize_transform = T.Compose([
            T.Resize((TARGET_HEIGHT, TARGET_WIDTH), antialias=True),  # height, width
        ])

    def __getitem__(self, idx):
        # Since length is 1, idx will always be 0
        if idx >= self.length:
            raise IndexError("Index out of range for single-video dataset")
            
        video_name = self.video_names[idx]
        paths = self._get_paths(video_name)

        # Load left and right data
        left_rgb = self._load_rgb(paths['left']['rgb'])
        left_event = self._load_events(paths['left']['event'])
        right_rgb = self._load_rgb(paths['right']['rgb'])
        right_event = self._load_events(paths['right']['event'])

        # Helper for frame-wise transform
        def apply_transform_to_sequence(sequence_tensor, transform_fn):
            if sequence_tensor.ndim == 3:
                return transform_fn(sequence_tensor)
            transformed_frames = [transform_fn(sequence_tensor[t]) for t in range(sequence_tensor.shape[0])]
            return torch.stack(transformed_frames, dim=0)

        left_rgb = apply_transform_to_sequence(left_rgb, self.transform_rgb)
        right_rgb = apply_transform_to_sequence(right_rgb, self.transform_rgb)
        left_event = apply_transform_to_sequence(left_event, self.transforms_evs)
        right_event = apply_transform_to_sequence(right_event, self.transforms_evs)

        # Center crop for inference
        left_pixel_values, left_events = self.crop_center_patch(left_rgb, left_event, random_crop=False)
        right_pixel_values, right_events = self.crop_center_patch(right_rgb, right_event, random_crop=False)

        # Select specific frames/events
        frame1 = left_pixel_values[0]             # first frame of left RGB
        frame2 = left_pixel_values[-1]            # last frame of left RGB
        evs = right_events                        # right events

        frame1_resized = self.resize_transform(frame1)
        frame2_resized = self.resize_transform(frame2)
        
        return dict(
            frame1=frame1_resized,
            frame2=frame2_resized,
            evs=evs,
            video_name=video_name
        )


def tensor_to_pillow(tensor, save_path):
    image_data = tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()/2.0 + 0.5
    image_data = image_data /np.max(image_data) * 255
    image_data = image_data.astype("uint8")
    pil_image = Image.fromarray(image_data)
    pil_image.save(save_path)
    return pil_image

def main(args):
    accelerator = Accelerator(
        mixed_precision="fp16")
    
    if accelerator.is_main_process:
        print(f"Using {accelerator.num_processes} GPUs for inference")
        print(f"Process index: {accelerator.process_index}")
        print("=" * 50)

    # Load scheduler
    noise_scheduler = EulerDiscreteScheduler.from_pretrained(
        args.pretrained_model_name_or_path, 
        subfolder="scheduler"
    )
    
    # Load pipeline
    if accelerator.is_main_process:
        print("Loading base pipeline...")
    
    pipe = EVSFrameInterpolationWithNoiseInjectionPipeline.from_pretrained(
        args.pretrained_model_name_or_path, 
        scheduler=noise_scheduler,
        variant="fp16",
        torch_dtype=torch.float16, 
    )
    
    # Load finetuned UNet
    if accelerator.is_main_process:
        print(f"Loading finetuned UNet from {args.checkpoint_dir}...")
    
    finetuned_unet = UNetSpatioTemporalConditionModel.from_pretrained(
        args.checkpoint_dir,
        subfolder="unet",
        torch_dtype=torch.float16,
    ) 

    finetuned_state_dict = finetuned_unet.state_dict()
    pipe.unet.load_state_dict(finetuned_state_dict)
    
    del finetuned_unet
    torch.cuda.empty_cache()

    if accelerator.is_main_process:
        print("UNet loaded successfully")

    # Move pipeline components to device
    pipe = pipe.to(accelerator.device)
    
    # --- FIX 1: Get num_frames BEFORE wrapping ---
    # Get the num_frames from the config *before* wrapping the unet
    num_frames = pipe.unet.config.num_frames
    if accelerator.is_main_process:
        print(f"UNet num_frames retrieved: {num_frames}")
    # --- END FIX ---
    
    # Prepare UNet with Accelerator for distributed inference
    pipe.unet = accelerator.prepare(pipe.unet)
    
    if accelerator.is_main_process:
        print("Model prepared for distributed inference")
        print("=" * 50)
    
    # Setup generator
    generator = torch.Generator(device=accelerator.device)
    if args.seed is not None:
        # Different seed for each GPU to add diversity
        generator = generator.manual_seed(args.seed + accelerator.process_index)
        
    # Load dataset
    if accelerator.is_main_process:
        print("Loading dataset...")
    
    dataset = StereoEventTestDataset(args.data_dir)
    
    if accelerator.is_main_process:
        print(f"Found {len(dataset)} videos")
    
    # Create dataloader
    dataloader = DataLoader(
        dataset, 
        batch_size=1, 
        shuffle=False, 
        num_workers=4,
        pin_memory=True
    )
    
    # Prepare dataloader with Accelerator (splits data across GPUs)
    dataloader = accelerator.prepare(dataloader)
    
    if accelerator.is_main_process:
        print(f"Processing {len(dataset)} samples across {accelerator.num_processes} GPUs")
        print(f"Each GPU will process approximately {len(dataloader)} samples")
        print("=" * 50)
    
    # Process videos
    for i, batch in enumerate(dataloader):
        # Calculate global index
        global_idx = i * accelerator.num_processes + accelerator.process_index
        
        video_name = batch['video_name'][0] if isinstance(batch['video_name'], (list, tuple)) else batch['video_name']
        
        if accelerator.is_main_process or True:  # Print from all processes
            print(f"[GPU {accelerator.process_index}] Processing batch {i} - Video: {video_name}")
            print(f"[GPU {accelerator.process_index}] Frame1 shape: {batch['frame1'].shape}")
            print(f"[GPU {accelerator.process_index}] Frame2 shape: {batch['frame2'].shape}")
            print(f"[GPU {accelerator.process_index}] Events shape: {batch['evs'].shape}")
        
        # Move batch to device if not already there
        frame1 = batch["frame1"].to(accelerator.device)
        frame2 = batch["frame2"].to(accelerator.device)
        evs = batch["evs"].to(accelerator.device)
        
        # Run inference
        with torch.no_grad():
            try:
                output = pipe(
                    image1=frame1, 
                    image2=frame2,
                    evs=evs, 
                    height=frame1.shape[-2], 
                    width=frame1.shape[-1],
                    
                    # --- FIX 2: Pass num_frames explicitly ---
                    num_frames=num_frames,
                    # --- END FIX ---
                    
                    num_inference_steps=args.num_inference_steps, 
                    generator=generator,
                    weighted_average=args.weighted_average,
                    noise_injection_steps=args.noise_injection_steps,
                    noise_injection_ratio=args.noise_injection_ratio,
                )
                frames = output.frames[0]
                
                # Save output
                os.makedirs(args.out_path, exist_ok=True)
                save_path = os.path.join(
                    args.out_path, 
                    f"{video_name}_gpu{accelerator.process_index}.mp4"
                )
                
                export_to_video(frames, save_path, fps=args.fps)
                
                print(f"[GPU {accelerator.process_index}] Saved video to: {save_path}")
                
            except Exception as e:
                print(f"[GPU {accelerator.process_index}] Error processing {video_name}: {str(e)}")
                import traceback
                traceback.print_exc()
    
    # Wait for all processes to finish
    accelerator.wait_for_everyone()
    
    if accelerator.is_main_process:
        print("=" * 50)
        print("All videos processed successfully!")
        print(f"Output saved to: {args.out_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrained_model_name_or_path", type=str, default="stabilityai/stable-video-diffusion-img2vid")
    parser.add_argument("--checkpoint_dir", type=str, default ='/data/venkateswara_lab/frame_interpollation/trained_models_full_recent/checkpoint-9500')
    parser.add_argument('--out_path', type=str, default="/data/venkateswara_lab/frame_interpollation/code/examples")
    parser.add_argument('--seed', type=int, default=8)
    parser.add_argument('--num_inference_steps', type=int, default=50)
    parser.add_argument('--weighted_average', action='store_true')
    parser.add_argument('--noise_injection_steps', type=int, default=0)
    parser.add_argument('--noise_injection_ratio', type=float, default=0.5)
    parser.add_argument('--device', type=str, default='cuda:0') # Note: Accelerator handles device placement
    
    # --- FIX: Added missing arguments ---
    parser.add_argument('--data_dir', type=str, default="/data/venkateswara_lab/frame_interpollation/data")
    parser.add_argument('--fps', type=int, default=10)
    # --- END FIX ---
    
    # These args seem unused by `main` but are kept from your original
    parser.add_argument('--frames_dirs', type=str, default='cuda:0')
    parser.add_argument('--event_filter', type=str, default=None)
    parser.add_argument('--skip_sampling_rate', type=int, default=1)
    
    args = parser.parse_args()
    
    # --- FIX: Create the out_path directory itself ---
    os.makedirs(args.out_path, exist_ok=True)
    
    # --- FIX: Added call to main() ---
    main(args)