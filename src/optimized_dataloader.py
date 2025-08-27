import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
import psutil
import gc
from typing import Optional, Tuple, Dict, Any
from optimized_dataset import DiarizationDataset, collate_fn


class MemoryAwareDataLoader(DataLoader):
    """DataLoader with dynamic memory management and adaptive batching."""
    
    def __init__(self, dataset, batch_size=16, shuffle=True, num_workers=4, 
                 pin_memory=True, memory_threshold=0.8, adaptive_batch=True,
                 distributed=False, **kwargs):
        
        self.memory_threshold = memory_threshold
        self.adaptive_batch = adaptive_batch
        self.base_batch_size = batch_size
        self.current_batch_size = batch_size
        self.distributed = distributed
        
        # Create sampler
        if distributed:
            sampler = DistributedSampler(dataset)
            shuffle = False
        else:
            sampler = RandomSampler(dataset) if shuffle else SequentialSampler(dataset)
        
        super().__init__(
            dataset=dataset,
            batch_size=batch_size,
            sampler=sampler,
            num_workers=num_workers,
            collate_fn=collate_fn,
            pin_memory=pin_memory,
            drop_last=True,  # Ensure consistent batch sizes
            **kwargs
        )
        
    def __iter__(self):
        """Override to add memory management."""
        for batch in super().__iter__():
            # Check memory usage and adjust batch size if needed
            if self.adaptive_batch:
                self._adjust_batch_size()
            
            # Clear cache periodically
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            yield batch
    
    def _adjust_batch_size(self):
        """Dynamically adjust batch size based on memory usage."""
        if torch.cuda.is_available():
            gpu_memory = torch.cuda.get_device_properties(0).total_memory
            gpu_allocated = torch.cuda.memory_allocated()
            gpu_usage = gpu_allocated / gpu_memory
            
            if gpu_usage > self.memory_threshold:
                # Reduce batch size
                new_batch_size = max(1, self.current_batch_size // 2)
                if new_batch_size != self.current_batch_size:
                    print(f"Reducing batch size: {self.current_batch_size} -> {new_batch_size}")
                    self.current_batch_size = new_batch_size
            elif gpu_usage < 0.5 and self.current_batch_size < self.base_batch_size:
                # Increase batch size
                new_batch_size = min(self.base_batch_size, self.current_batch_size * 2)
                if new_batch_size != self.current_batch_size:
                    print(f"Increasing batch size: {self.current_batch_size} -> {new_batch_size}")
                    self.current_batch_size = new_batch_size
        
        # Also check system RAM
        ram_usage = psutil.virtual_memory().percent / 100.0
        if ram_usage > self.memory_threshold:
            gc.collect()


class AccumulatingDataLoader:
    """DataLoader wrapper that accumulates gradients for effective larger batch sizes."""
    
    def __init__(self, dataloader: DataLoader, accumulation_steps: int = 2):
        self.dataloader = dataloader
        self.accumulation_steps = accumulation_steps
        self.step_count = 0
        
    def __iter__(self):
        accumulated_batch = []
        
        for batch in self.dataloader:
            accumulated_batch.append(batch)
            self.step_count += 1
            
            if len(accumulated_batch) == self.accumulation_steps:
                # Yield accumulated batch
                yield self._combine_batches(accumulated_batch)
                accumulated_batch = []
        
        # Yield remaining batch if any
        if accumulated_batch:
            yield self._combine_batches(accumulated_batch)
    
    def _combine_batches(self, batches):
        """Combine multiple batches into one."""
        if len(batches) == 1:
            return batches[0]
        
        combined = {}
        for key in batches[0].keys():
            if isinstance(batches[0][key], torch.Tensor):
                combined[key] = torch.cat([batch[key] for batch in batches], dim=0)
            elif isinstance(batches[0][key], list):
                combined[key] = []
                for batch in batches:
                    combined[key].extend(batch[key])
            else:
                combined[key] = batches[0][key]  # Take first value for non-tensor/list items
        
        return combined
    
    def __len__(self):
        return (len(self.dataloader) + self.accumulation_steps - 1) // self.accumulation_steps


def create_optimized_dataloaders(
    audio_dir: str,
    rttm_dir: str,
    batch_size: int = 16,
    train_split: float = 0.8,
    num_workers: int = 4,
    memory_threshold: float = 0.8,
    adaptive_batch: bool = True,
    accumulation_steps: Optional[int] = None,
    distributed: bool = False,
    **dataset_kwargs
) -> Tuple[DataLoader, DataLoader]:
    """
    Create optimized train and validation dataloaders with memory management.
    
    Args:
        audio_dir: Path to audio files
        rttm_dir: Path to RTTM annotation files
        batch_size: Base batch size
        train_split: Training split ratio
        num_workers: Number of data loading workers
        memory_threshold: Memory usage threshold for adaptive batching
        adaptive_batch: Whether to use adaptive batch sizing
        accumulation_steps: Steps for gradient accumulation (None to disable)
        distributed: Whether to use distributed training
        **dataset_kwargs: Additional arguments for dataset
        
    Returns:
        Tuple of (train_loader, val_loader)
    """
    
    # Create full dataset
    full_dataset = DiarizationDataset(audio_dir, rttm_dir, **dataset_kwargs)
    
    if len(full_dataset) == 0:
        raise ValueError("No valid audio-RTTM pairs found!")
    
    # Train/val split
    train_size = int(train_split * len(full_dataset))
    val_size = len(full_dataset) - train_size
    
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    # Create memory-aware dataloaders
    train_loader = MemoryAwareDataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        memory_threshold=memory_threshold,
        adaptive_batch=adaptive_batch,
        distributed=distributed,
        pin_memory=True
    )
    
    val_loader = MemoryAwareDataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        memory_threshold=memory_threshold,
        adaptive_batch=False,  # No adaptive batching for validation
        distributed=distributed,
        pin_memory=True
    )
    
    # Wrap with gradient accumulation if requested
    if accumulation_steps and accumulation_steps > 1:
        train_loader = AccumulatingDataLoader(train_loader, accumulation_steps)
    
    return train_loader, val_loader


class StreamingDataLoader:
    """Streaming dataloader for very large datasets that don't fit in memory."""
    
    def __init__(self, audio_dir: str, rttm_dir: str, batch_size: int = 16,
                 max_segments_per_file: int = 100, **dataset_kwargs):
        self.audio_dir = audio_dir
        self.rttm_dir = rttm_dir
        self.batch_size = batch_size
        self.max_segments_per_file = max_segments_per_file
        self.dataset_kwargs = dataset_kwargs
        
        # Get list of audio files
        import os
        self.audio_files = []
        if os.path.exists(audio_dir):
            for filename in os.listdir(audio_dir):
                if filename.endswith('.wav'):
                    base_name = filename.replace('.wav', '')
                    rttm_path = os.path.join(rttm_dir, f"{base_name}.rttm")
                    if os.path.exists(rttm_path):
                        self.audio_files.append((
                            os.path.join(audio_dir, filename),
                            rttm_path
                        ))
        
        print(f"Streaming dataloader initialized with {len(self.audio_files)} files")
    
    def __iter__(self):
        """Stream batches from files on-demand."""
        for audio_path, rttm_path in self.audio_files:
            # Create temporary dataset for this file
            temp_audio_dir = '/tmp/single_file_audio'
            temp_rttm_dir = '/tmp/single_file_rttm'
            
            # Create directories and copy files temporarily
            import os, shutil
            os.makedirs(temp_audio_dir, exist_ok=True)
            os.makedirs(temp_rttm_dir, exist_ok=True)
            
            shutil.copy2(audio_path, temp_audio_dir)
            shutil.copy2(rttm_path, temp_rttm_dir)
            
            try:
                # Create dataset for this file
                file_dataset = DiarizationDataset(
                    temp_audio_dir, temp_rttm_dir,
                    max_segments=self.max_segments_per_file,
                    **self.dataset_kwargs
                )
                
                if len(file_dataset) == 0:
                    continue
                
                # Create dataloader for this file
                file_loader = DataLoader(
                    file_dataset,
                    batch_size=self.batch_size,
                    shuffle=True,
                    collate_fn=collate_fn,
                    num_workers=0  # Use 0 workers for streaming
                )
                
                # Yield batches from this file
                for batch in file_loader:
                    yield batch
                
            finally:
                # Cleanup
                shutil.rmtree(temp_audio_dir, ignore_errors=True)
                shutil.rmtree(temp_rttm_dir, ignore_errors=True)
                
                # Clear cache
                if hasattr(file_dataset, 'feature_cache'):
                    file_dataset.feature_cache.clear()
                del file_dataset
                gc.collect()


if __name__ == "__main__":
    # Test optimized dataloaders
    audio_dir = "/path/to/audio/files"
    rttm_dir = "/path/to/rttm/files"
    
    try:
        # Test memory-aware dataloader
        train_loader, val_loader = create_optimized_dataloaders(
            audio_dir, rttm_dir,
            batch_size=8,
            max_segments=50,
            adaptive_batch=True,
            accumulation_steps=2
        )
        
        print(f"Train batches: {len(train_loader)}")
        print(f"Val batches: {len(val_loader)}")
        
        # Test one batch
        for batch in train_loader:
            print(f"Batch features shape: {batch['features'].shape}")
            print(f"Batch VAD labels shape: {batch['vad_labels'].shape}")
            print(f"Memory usage: {psutil.virtual_memory().percent:.1f}%")
            break
            
    except Exception as e:
        print(f"Error testing optimized dataloaders: {e}")
        
        # Test streaming dataloader as fallback
        print("Testing streaming dataloader...")
        streaming_loader = StreamingDataLoader(
            audio_dir, rttm_dir, batch_size=4, max_segments_per_file=20
        )
        
        batch_count = 0
        for batch in streaming_loader:
            print(f"Streaming batch {batch_count}: {batch['features'].shape}")
            batch_count += 1
            if batch_count >= 3:  # Test first 3 batches
                break