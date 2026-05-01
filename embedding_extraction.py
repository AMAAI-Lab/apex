import json
import os
import argparse
import torch
from transformers import AutoProcessor, AutoModel
import soundfile as sf
from torch import nn
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm
from multiprocessing import current_process, Manager, Process
import logging


# MODEL CONFIG

MODEL_NAME    = "m-a-p/MERT-v1-95M"
LAYER_INDICES = [2, 5, 8, -1]
SEGMENT_SEC   = 30
SEED          = 42


# LOGGING

logging.basicConfig(
    filename = "embedding_extraction.log",
    level    = logging.INFO,
    format   = "%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger()
logger.addHandler(logging.StreamHandler())



# INIT PROCESS

def init_process(gpu_id):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    global device, processor, model, aggregator

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    processor = AutoProcessor.from_pretrained(MODEL_NAME, trust_remote_code=True)
    model     = AutoModel.from_pretrained(MODEL_NAME, trust_remote_code=True)
    model.eval()
    model.to(device)

    torch.manual_seed(SEED)
    aggregator = nn.Conv1d(
        in_channels  = len(LAYER_INDICES),
        out_channels = 1,
        kernel_size  = 1
    ).to(device)

    logger.info(f"Process {current_process().name} initialized on GPU {gpu_id}")



# PROCESS AUDIO FILE

def process_audio_file(audio_path, audio_id, score_streams, score_likes, platform):
    try:
        waveform, sr = sf.read(audio_path, dtype="float32")
        waveform     = torch.from_numpy(waveform)
        if len(waveform.shape) > 1 and waveform.shape[1] > 1:
            waveform = waveform.mean(dim=1)
    except Exception as e:
        logger.warning(f"Failed to load audio {audio_path}: {e}")
        return None

    try:
        waveform  = waveform.to(device)
        target_sr = processor.sampling_rate

        if sr != target_sr:
            import torchaudio.functional as F
            waveform = F.resample(waveform, sr, target_sr)

        segment_len     = SEGMENT_SEC * target_sr
        segment_records = []

        for i, start in enumerate(range(0, waveform.shape[0], segment_len)):
            segment = waveform[start:start + segment_len]
            if segment.numel() == 0:
                break
            if segment.shape[0] < segment_len:
                pad_len = segment_len - segment.shape[0]
                segment = torch.nn.functional.pad(segment, (0, pad_len))

            inputs = processor(segment.cpu().numpy(), sampling_rate=target_sr, return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = model(**inputs, output_hidden_states=True)

            all_hidden     = torch.stack([outputs.hidden_states[i].mean(dim=1) for i in LAYER_INDICES])
            all_hidden     = all_hidden.squeeze(1)
            pooled_segment = aggregator(all_hidden.unsqueeze(0)).squeeze()

            segment_records.append({
                "audio_id"         : audio_id,
                "segment_id"       : f"{audio_id}_{i+1}",
                "score_streams"    : score_streams,
                "score_likes"      : score_likes,
                "platform"         : platform,
                "segment_embedding": pooled_segment.detach().cpu().numpy()
            })

            del segment, inputs, outputs, all_hidden, pooled_segment

        del waveform
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return segment_records

    except Exception as e:
        logger.error(f"Error processing {audio_id}: {e}")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return None



# SAVE SHARD

def save_shard(records, gpu_id, shard_counter, parquet_folder, suffix=""):
    if not records:
        return shard_counter
    shard_counter += 1
    shard_path = os.path.join(parquet_folder, f"gpu{gpu_id}_shard_{shard_counter:04d}{suffix}.parquet")
    pq.write_table(pa.Table.from_pandas(pd.DataFrame(records)), shard_path)
    logger.info(f"Saved {shard_path} with {len(records)} records")
    return shard_counter



# WORKER

def worker_process(gpu_id, task_queue, progress_queue, checkpoint_queue, batch_size, parquet_folder, audio_folder, records_per_shard, partial_save_interval):
    init_process(gpu_id)

    shard_counter   = 0
    current_records = []
    missing_files   = []
    total_processed = 0

    while True:
        batch = []
        for _ in range(batch_size):
            try:
                song = task_queue.get(timeout=0.1)
                if song is None:
                    break
                batch.append(song)
            except:
                break

        if not batch:
            break

        logger.info(f"GPU{gpu_id}: Processing batch of {len(batch)} songs")

        for song in batch:
            audio_id      = song["id"]
            score_streams = song["score_streams"]
            score_likes   = song["score_likes"]
            platform      = song["platform"]
            audio_path    = os.path.join(audio_folder, f"{audio_id}.mp3")

            if not os.path.exists(audio_path):
                logger.warning(f"GPU{gpu_id}: Audio not found: {audio_path}")
                missing_files.append(audio_path)
                progress_queue.put(1)
                checkpoint_queue.put(audio_id)
                continue

            segment_records = process_audio_file(audio_path, audio_id, score_streams, score_likes, platform)
            if segment_records is None:
                missing_files.append(audio_path)
                progress_queue.put(1)
                checkpoint_queue.put(audio_id)
                continue

            current_records.extend(segment_records)
            total_processed += 1

            while len(current_records) >= records_per_shard:
                shard_data      = current_records[:records_per_shard]
                current_records = current_records[records_per_shard:]
                shard_counter   = save_shard(shard_data, gpu_id, shard_counter, parquet_folder)

            if len(current_records) >= partial_save_interval:
                shard_counter   = save_shard(current_records, gpu_id, shard_counter, parquet_folder, suffix="_partial")
                current_records = []

            progress_queue.put(1)
            checkpoint_queue.put(audio_id)

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        logger.info(f"GPU{gpu_id}: Completed batch. Total processed: {total_processed}")

    if current_records:
        save_shard(current_records, gpu_id, shard_counter, parquet_folder, suffix="_final")

    if missing_files:
        missing_log = os.path.join(parquet_folder, f"missing_gpu{gpu_id}.txt")
        with open(missing_log, "w") as f:
            for path in missing_files:
                f.write(path + "\n")
        logger.info(f"GPU{gpu_id}: Saved missing files to {missing_log}")

    logger.info(f"GPU{gpu_id}: Finished. Processed {total_processed} songs.")



# MERGE PARTIAL SHARDS

def merge_partial_shards(parquet_folder, gpu_id, records_per_shard):
    import glob
    partial_files = sorted(glob.glob(os.path.join(parquet_folder, f"gpu{gpu_id}_shard_*_partial.parquet")))
    final_files   = sorted(glob.glob(os.path.join(parquet_folder, f"gpu{gpu_id}_shard_*_final.parquet")))
    all_partial   = partial_files + final_files

    if not all_partial:
        return

    shard_counter = max([
        int(os.path.basename(f).split("_")[2])
        for f in glob.glob(os.path.join(parquet_folder, f"gpu{gpu_id}_shard_*.parquet"))
        if "_partial" not in f and "_final" not in f
    ] + [0])

    temp_records = []
    for pf in all_partial:
        table = pq.read_table(pf)
        for df_chunk in table.to_pandas(chunksize=1000):
            temp_records.extend(df_chunk.to_dict(orient="records"))
            while len(temp_records) >= records_per_shard:
                shard_counter += 1
                shard_path     = os.path.join(parquet_folder, f"gpu{gpu_id}_shard_{shard_counter:04d}.parquet")
                pq.write_table(pa.Table.from_pandas(pd.DataFrame(temp_records[:records_per_shard])), shard_path)
                temp_records   = temp_records[records_per_shard:]
                logger.info(f"Merged partial -> {shard_path}")
        os.remove(pf)

    if temp_records:
        shard_counter += 1
        shard_path     = os.path.join(parquet_folder, f"gpu{gpu_id}_shard_{shard_counter:04d}.parquet")
        pq.write_table(pa.Table.from_pandas(pd.DataFrame(temp_records)), shard_path)
        logger.info(f"Merged partial -> {shard_path}")



# MAIN

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="APEX Embedding Extraction")
    parser.add_argument("--jsonl_file",            type=str, required=True,  help="Path to input JSONL file")
    parser.add_argument("--audio_folder",          type=str, required=True,  help="Path to folder containing MP3 files")
    parser.add_argument("--parquet_folder",        type=str, required=True,  help="Path to output parquet folder")
    parser.add_argument("--num_gpus",              type=int, default=4,      help="Number of GPUs to use")
    parser.add_argument("--songs_per_batch",       type=int, default=50,     help="Songs per GPU batch")
    parser.add_argument("--records_per_shard",     type=int, default=10000,  help="Records per parquet shard")
    parser.add_argument("--partial_save_interval", type=int, default=1000,   help="Save partial shard every N records")
    args = parser.parse_args()

    os.makedirs(args.parquet_folder, exist_ok=True)
    checkpoint_file = os.path.join(args.parquet_folder, "processed_songs.json")

    # Load checkpoint
    if os.path.exists(checkpoint_file):
        with open(checkpoint_file, "r") as f:
            processed_ids = set(json.load(f))
    else:
        processed_ids = set()

    # Load songs
    with open(args.jsonl_file, "r") as f:
        songs = [json.loads(line) for line in f if json.loads(line)["id"] not in processed_ids]

    logger.info(f"Total songs to process: {len(songs):,}")

    # Queues
    manager          = Manager()
    task_queue       = manager.Queue()
    progress_queue   = manager.Queue()
    checkpoint_queue = manager.Queue()

    for song in songs:
        task_queue.put(song)
    for _ in range(args.num_gpus):
        task_queue.put(None)

    # Start workers
    workers = []
    for gpu_id in range(args.num_gpus):
        p = Process(
            target = worker_process,
            args   = (
                gpu_id, task_queue, progress_queue, checkpoint_queue,
                args.songs_per_batch, args.parquet_folder, args.audio_folder,
                args.records_per_shard, args.partial_save_interval
            )
        )
        p.start()
        workers.append(p)
        logger.info(f"Started worker for GPU {gpu_id}")

    # Progress + checkpoint
    overall_bar        = tqdm(total=len(songs), desc="Overall Progress")
    checkpoint_buffer  = []

    while any(w.is_alive() for w in workers):
        while not progress_queue.empty():
            overall_bar.update(progress_queue.get())
        while not checkpoint_queue.empty():
            checkpoint_buffer.append(checkpoint_queue.get())
        if len(checkpoint_buffer) >= 100:
            processed_ids.update(checkpoint_buffer)
            with open(checkpoint_file, "w") as f:
                json.dump(list(processed_ids), f)
            checkpoint_buffer = []

    for w in workers:
        w.join()

    if checkpoint_buffer:
        processed_ids.update(checkpoint_buffer)
        with open(checkpoint_file, "w") as f:
            json.dump(list(processed_ids), f)

    overall_bar.close()
    logger.info("All workers finished!")

    # Merge partial shards
    logger.info("Merging partial shards...")
    for gpu_id in range(args.num_gpus):
        merge_partial_shards(args.parquet_folder, gpu_id, args.records_per_shard)

    logger.info("Done!")