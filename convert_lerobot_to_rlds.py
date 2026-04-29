"""
convert_lerobot_to_rlds.py

Convert LeRobot v2.x datasets (parquet + video) to RLDS TFRecord format
for training with MemoryVLA and RoboVLMs (both use the OpenVLA/OXE RLDS pipeline).

Supports cup_full and pizza_v2 datasets, or any LeRobot v2.x format dataset.

Usage:
    # Convert cup_full dataset:
    python convert_lerobot_to_rlds.py \
        --input_dir /data_16T/lerobot_openx/cup_full \
        --output_dir /home/v-wenhuitan/TCD/MemoryVLA/data/custom_finetuning \
        --primary_image observation.images.right_rgb \
        --secondary_image observation.images.top_rgb

    # Convert pizza_v2 dataset:
    python convert_lerobot_to_rlds.py \
        --input_dir /data_16T/lerobot_openx/pizza_v2 \
        --output_dir /home/v-wenhuitan/TCD/MemoryVLA/data/custom_finetuning \
        --primary_image observation.images.primary \
        --secondary_image observation.images.secondary

    # Then register in configs:
    # MemoryVLA: already has "custom_finetuning" entry in OXE_DATASET_CONFIGS
    # RoboVLMs: add "custom_finetuning" to train_dataset.data_mix in JSON config
"""

import os
import sys
import json
import argparse
import struct
import numpy as np
from pathlib import Path
from collections import defaultdict

import cv2
import tensorflow as tf
import pyarrow.parquet as pq


def load_episodes_from_parquet(input_dir):
    """Load all episode data from LeRobot parquet files."""
    data_dir = Path(input_dir) / "data"
    episodes = defaultdict(list)

    # Find all chunk dirs
    chunk_dirs = sorted([d for d in data_dir.iterdir() if d.is_dir() and d.name.startswith("chunk-")])
    if not chunk_dirs:
        # No chunk dirs — parquet files directly in data/
        chunk_dirs = [data_dir]

    for chunk_dir in chunk_dirs:
        parquet_files = sorted(chunk_dir.glob("episode_*.parquet"))
        for pf in parquet_files:
            table = pq.read_table(pf)
            df = table.to_pandas()
            for _, row in df.iterrows():
                ep_idx = row["episode_index"]
                episodes[ep_idx].append(row.to_dict())

    # Sort each episode by frame_index
    for ep_idx in episodes:
        episodes[ep_idx].sort(key=lambda r: r["frame_index"])

    print(f"Loaded {len(episodes)} episodes from {input_dir}")
    return episodes


def load_tasks(input_dir):
    """Load task descriptions from tasks.jsonl."""
    tasks_file = Path(input_dir) / "meta" / "tasks.jsonl"
    tasks = {}
    if tasks_file.exists():
        with open(tasks_file) as f:
            for line in f:
                entry = json.loads(line.strip())
                tasks[entry["task_index"]] = entry["task"]
    return tasks


def load_stats(input_dir):
    """Load dataset statistics from stats.json."""
    stats_file = Path(input_dir) / "meta" / "stats.json"
    if stats_file.exists():
        with open(stats_file) as f:
            return json.load(f)
    return None


def extract_video_frame(video_path, frame_index):
    """Extract a single frame from a video file."""
    cap = cv2.VideoCapture(str(video_path))
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        print(f"WARNING: Failed to read frame {frame_index} from {video_path}")
        return None
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return frame


def encode_image_as_jpeg(image_np, quality=95):
    """Encode a numpy RGB image as JPEG bytes."""
    image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
    _, encoded = cv2.imencode('.jpg', image_bgr, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return encoded.tobytes()


def find_video_path(input_dir, episode_index, video_key):
    """Find the video file path for a given episode and video key."""
    videos_dir = Path(input_dir) / "videos"

    # Try chunk-based structure first
    chunk_id = episode_index // 1000
    chunk_dir = videos_dir / f"chunk-{chunk_id:03d}" / video_key
    video_file = chunk_dir / f"episode_{episode_index:06d}.mp4"

    if video_file.exists():
        return video_file

    # Try flat structure
    video_file = videos_dir / video_key / f"episode_{episode_index:06d}.mp4"
    if video_file.exists():
        return video_file

    return None


def create_rlds_trajectory(episode_frames, video_cache, primary_image_key,
                           secondary_image_key=None, wrist_image_key=None,
                           image_size=(224, 224)):
    """Create an RLDS trajectory from a list of episode frames.

    RLDS trajectory structure:
        steps: list of step dicts with:
            observation: {image (JPEG bytes), state (float array)}
            action: float array (7 dim)
            language_instruction: string
            is_terminal: bool
            is_last: bool
            timestamp: float
    """
    steps = []

    for i, frame_data in enumerate(episode_frames):
        ep_idx = frame_data["episode_index"]
        frame_idx = frame_data["frame_index"]
        task_idx = frame_data.get("task_index", 0)

        # --- Extract image from video ---
        image_bytes = None
        if primary_image_key in video_cache and ep_idx in video_cache[primary_image_key]:
            image_np = video_cache[primary_image_key][ep_idx].get(frame_idx)
            if image_np is not None:
                image_bytes = encode_image_as_jpeg(image_np)

        if image_bytes is None:
            # Fallback: create black image
            black = np.zeros((image_size[0], image_size[1], 3), dtype=np.uint8)
            image_bytes = encode_image_as_jpeg(black)

        # Secondary image (optional)
        secondary_image_bytes = None
        if secondary_image_key and secondary_image_key in video_cache \
                and ep_idx in video_cache[secondary_image_key]:
            sec_np = video_cache[secondary_image_key][ep_idx].get(frame_idx)
            if sec_np is not None:
                secondary_image_bytes = encode_image_as_jpeg(sec_np)

        # --- Extract action ---
        action = np.array(frame_data["action"], dtype=np.float32)
        if len(action) != 7:
            action = np.pad(action, (0, 7 - len(action)))[:7]

        # --- Extract state ---
        state = np.array(frame_data.get("observation.state", [0.0]*8), dtype=np.float32)

        # --- Language instruction ---
        language_instruction = frame_data.get("_language_instruction", "")

        # --- Terminal flag ---
        is_last = (i == len(episode_frames) - 1)
        is_terminal = is_last

        step = {
            "observation": {
                "image": image_bytes,
                "base_pose_tool_reached": state[:7].tolist(),
                "gripper_closed": [state[7] if len(state) > 7 else 0.0],
            },
            "action": action.tolist(),
            "language_instruction": language_instruction,
            "is_terminal": is_terminal,
            "is_last": is_last,
            "timestamp": [frame_data.get("timestamp", i / 30.0)],
        }

        if secondary_image_bytes:
            step["observation"]["image_1"] = secondary_image_bytes

        steps.append(step)

    return steps


def write_rlds_tfrecord(trajectories, output_path):
    """Write trajectories to a TFRecord file in RLDS format."""
    with tf.io.TFRecordWriter(str(output_path)) as writer:
        for traj in trajectories:
            # Serialize each trajectory as a sequence of steps
            feature_dict = {}
            step_features_list = []

            for step in traj["steps"]:
                step_features = {
                    "observation/image": tf.train.Feature(
                        bytes_list=tf.train.BytesList(value=[step["observation"]["image"]])
                    ),
                    "observation/base_pose_tool_reached": tf.train.Feature(
                        float_list=tf.train.FloatList(value=step["observation"]["base_pose_tool_reached"])
                    ),
                    "observation/gripper_closed": tf.train.Feature(
                        float_list=tf.train.FloatList(value=step["observation"]["gripper_closed"])
                    ),
                    "action": tf.train.Feature(
                        float_list=tf.train.FloatList(value=step["action"])
                    ),
                    "language_instruction": tf.train.Feature(
                        bytes_list=tf.train.BytesList(
                            value=[step["language_instruction"].encode("utf-8")]
                        )
                    ),
                    "is_terminal": tf.train.Feature(
                        int64_list=tf.train.Int64List(value=[int(step["is_terminal"])])
                    ),
                    "is_last": tf.train.Feature(
                        int64_list=tf.train.Int64List(value=[int(step["is_last"])])
                    ),
                    "timestamp": tf.train.Feature(
                        float_list=tf.train.FloatList(value=step["timestamp"])
                    ),
                }

                if "image_1" in step["observation"]:
                    step_features["observation/image_1"] = tf.train.Feature(
                        bytes_list=tf.train.BytesList(value=[step["observation"]["image_1"]])
                    )

                step_features_list.append(step_features)

            # Write each step as a separate TFRecord example
            for step_features in step_features_list:
                example = tf.train.Example(
                    features=tf.train.Features(feature=step_features)
                )
                writer.write(example.SerializeToString())


def load_videos_for_episodes(input_dir, episodes, video_keys, max_episodes=None):
    """Pre-load video frames for all episodes."""
    video_cache = {}

    ep_indices = sorted(episodes.keys())
    if max_episodes:
        ep_indices = ep_indices[:max_episodes]

    for vk in video_keys:
        video_cache[vk] = {}
        print(f"Loading videos for {vk}...")

        for ep_idx in ep_indices:
            video_path = find_video_path(input_dir, ep_idx, vk)
            if video_path is None:
                continue

            cap = cv2.VideoCapture(str(video_path))
            frames = {}
            frame_idx = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames[frame_idx] = frame_rgb
                frame_idx += 1
            cap.release()
            video_cache[vk][ep_idx] = frames
            print(f"  Episode {ep_idx}: {frame_idx} frames from {vk}")

    return video_cache


def write_dataset_metadata(output_dir, stats, num_episodes, num_steps):
    """Write dataset_info.json and features.json for RLDS."""
    output_path = Path(output_dir)

    # dataset_info.json
    dataset_info = {
        "name": "custom_finetuning",
        "splits": {
            "train": {
                "numShards": 1,
                "numExamples": num_steps,
                "numEpisodes": num_episodes,
            }
        },
        "version": "1.0.0",
    }
    with open(output_path / "dataset_info.json", "w") as f:
        json.dump(dataset_info, f, indent=2)

    # features.json
    features = {
        "steps": {
            "observation": {
                "image": {"dtype": "uint8", "shape": [224, 224, 3], "featureType": "IMAGE"},
                "base_pose_tool_reached": {"dtype": "float32", "shape": [7]},
                "gripper_closed": {"dtype": "float32", "shape": [1]},
            },
            "action": {"dtype": "float32", "shape": [7]},
            "language_instruction": {"dtype": "string"},
            "is_terminal": {"dtype": "bool"},
            "is_last": {"dtype": "bool"},
            "timestamp": {"dtype": "float32", "shape": [1]},
        },
        "episode_metadata": {
            "episode_id": {"dtype": "int64"},
        },
    }
    with open(output_path / "features.json", "w") as f:
        json.dump(features, f, indent=2)

    print(f"Metadata written to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Convert LeRobot v2.x dataset to RLDS TFRecord format"
    )
    parser.add_argument("--input_dir", type=str, required=True,
                        help="Path to LeRobot dataset (e.g., /data_16T/lerobot_openx/cup_full)")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Output RLDS directory (e.g., <data_root_dir>/custom_finetuning)")
    parser.add_argument("--primary_image", type=str, default="observation.images.right_rgb",
                        help="Video key for primary (third-person) image")
    parser.add_argument("--secondary_image", type=str, default=None,
                        help="Video key for secondary image (optional)")
    parser.add_argument("--wrist_image", type=str, default=None,
                        help="Video key for wrist camera image (optional)")
    parser.add_argument("--max_episodes", type=int, default=None,
                        help="Limit number of episodes to convert (for debugging)")
    parser.add_argument("--fps", type=int, default=30,
                        help="Dataset FPS for timestamp calculation")

    args = parser.parse_args()

    # Create output directory structure: <output_dir>/1.0.0/
    version_dir = Path(args.output_dir) / "1.0.0"
    version_dir.mkdir(parents=True, exist_ok=True)

    # Load episodes and tasks
    episodes = load_episodes_from_parquet(args.input_dir)
    tasks = load_tasks(args.input_dir)
    stats = load_stats(args.input_dir)

    # Determine video keys to load
    video_keys = [args.primary_image]
    if args.secondary_image:
        video_keys.append(args.secondary_image)
    if args.wrist_image:
        video_keys.append(args.wrist_image)

    # Load video frames
    video_cache = load_videos_for_episodes(
        args.input_dir, episodes, video_keys, args.max_episodes
    )

    # Assign language instructions to frames
    ep_indices = sorted(episodes.keys())
    if args.max_episodes:
        ep_indices = ep_indices[:args.max_episodes]

    total_steps = 0
    trajectories = []

    for ep_idx in ep_indices:
        ep_frames = episodes[ep_idx]

        # Determine language instruction for this episode
        task_idx = ep_frames[0].get("task_index", 0)
        language_instruction = tasks.get(task_idx, "")

        # Attach language instruction to each frame
        for frame in ep_frames:
            frame["_language_instruction"] = language_instruction

        # Create trajectory
        traj_steps = create_rlds_trajectory(
            ep_frames,
            video_cache,
            primary_image_key=args.primary_image,
            secondary_image_key=args.secondary_image,
            wrist_image_key=args.wrist_image,
        )

        trajectory = {
            "steps": traj_steps,
            "episode_id": ep_idx,
        }
        trajectories.append(trajectory)
        total_steps += len(traj_steps)
        print(f"Episode {ep_idx}: {len(traj_steps)} steps, task: '{language_instruction}'")

    # Write TFRecord
    tfrecord_path = version_dir / "train-00000-of-00001.tfrecord"
    print(f"Writing {total_steps} steps from {len(trajectories)} episodes to {tfrecord_path}")
    write_rlds_tfrecord(trajectories, tfrecord_path)

    # Write metadata
    write_dataset_metadata(version_dir, stats, len(trajectories), total_steps)

    print(f"\nConversion complete!")
    print(f"  Output: {version_dir}")
    print(f"  Episodes: {len(trajectories)}")
    print(f"  Steps: {total_steps}")
    print(f"\nTo use with MemoryVLA training:")
    print(f"  --data_root_dir {Path(args.output_dir).parent}")
    print(f"  --vla.data_mix custom_finetuning")
    print(f"\nTo use with RoboVLMs training:")
    print(f"  --train_dataset.data_root_dir {Path(args.output_dir).parent}")
    print(f"  --train_dataset.data_mix custom_finetuning")


if __name__ == "__main__":
    main()