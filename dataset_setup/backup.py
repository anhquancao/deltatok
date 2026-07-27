import os
import tarfile
import shutil
from pathlib import Path
import time
import argparse
from tqdm import tqdm
from multiprocessing import Pool


def create_tar_archive(source_dir, dest_dir, src_dir, dryrun=False):
    """
    Create a tar archive of the source directory and save it to the destination directory.
    Also records successful backups in backup.txt.

    Args:
        source_dir (str): Relative path to the source directory to be archived
        dest_dir (str): Path to the destination directory where the archive will be saved
        src_dir (str): Base source directory path
        dryrun (bool): If True, only simulate the operations without making changes
    """
    source_path = Path(os.path.join(src_dir, source_dir))
    if not source_path.exists() or not source_path.is_dir():
        print(f"Source directory {source_dir} does not exist or is not a directory")
        return
    # Create destination directory if it doesn't exist
    dest_path = Path(dest_dir)
    if not dryrun:
        dest_path.mkdir(parents=True, exist_ok=True)

    # Create archive name based on directory name (.tar.gz kept for continuity with the
    # existing backup set — contents are uncompressed, see tarfile.open(..., "w") below)
    archive_name = f"{source_path.name}.tar.gz"
    archive_path = dest_path / archive_name

    # Skip if archive already exists
    # if archive_path.exists():
    #     print(f"Archive already exists: {archive_path}, skipping...")
    #     return

    if dryrun:
        print(f"[DRYRUN] Would create archive {archive_path} from {source_path}")
        print(f"[DRYRUN] Would write to backup.txt: {source_dir}")
        return

    start_time = time.time()
    print(f"Creating archive {archive_path} from {source_path}...")

    # Collect all files first for progress bar
    all_files = []
    for root, _, files in os.walk(source_path):
        for file in files:
            all_files.append(os.path.join(root, file))

    # "w" = uncompressed: npz/png payloads are already compressed, gzip only costs time
    with tarfile.open(archive_path, "w") as tar:
        for file_path in tqdm(all_files, desc=f"Archiving {source_path.name}", leave=False):
            arcname = os.path.relpath(file_path, source_path.parent)
            try:
                tar.add(file_path, arcname=arcname)
            except OSError as e:
                print(f"Warning: Skipping file {file_path} due to error: {e}")

    elapsed_time = time.time() - start_time
    print(f"Archive created successfully: {archive_path} (took {elapsed_time:.2f} seconds)")

    # Write successful backup to backup.txt (relative path)
    # try:
    #     with open(backup_file, 'a') as f:
    #         # Use file locking to prevent concurrent writes
    #         fcntl.flock(f, fcntl.LOCK_EX)
    #         f.write(f"{source_dir}\n")
    #         fcntl.flock(f, fcntl.LOCK_UN)
    # except Exception as e:
    #     print(f"Warning: Could not write to backup.txt: {e}")

def extract_tar_archive(archive_path, target_dir, dryrun=False):
    """
    Extract a tar archive into the target directory.

    Args:
        archive_path (str): Path to the .tar or .tar.gz archive
        target_dir (str): Directory where the archive contents will be extracted
        dryrun (bool): If True, only simulate extraction
    """
    if dryrun:
        print(f"[DRYRUN] Would extract {archive_path} to {target_dir}")
        return 0

    start_time = time.time()

    archive_name = os.path.basename(archive_path)
    print(f"Extracting {archive_path} to {target_dir}...")
    with tarfile.open(archive_path, 'r:*') as tar:
        # Retrieve members before extraction so we can update their timestamps afterwards
        members = tar.getmembers()
        tar.extractall(path=target_dir)

        # Update timestamps of the extracted files/directories to the current time
        current_time = time.time()
        for member in members:
            extracted_path = os.path.join(target_dir, member.name)
            try:
                os.utime(extracted_path, (current_time, current_time))
            except (FileNotFoundError, PermissionError):
                # Skip entries that cannot be updated (e.g. special files or missing paths)
                pass

    # Also stamp the rest of the scene dir: companions written after the backup (e.g.
    # *.infinidepth.png) are not archive members and would keep a purge-eligible atime.
    # One archive == one scene dir, so parallel extractions never overlap here.
    scene_dir = os.path.join(target_dir, os.path.basename(archive_path).split('.tar')[0])
    for root, dirs, files in os.walk(scene_dir):
        for name in dirs + files:
            try:
                os.utime(os.path.join(root, name), (current_time, current_time))
            except OSError:
                pass
    
    num_files = len(members)
    elapsed_time = time.time() - start_time
    print(f"Archive extracted successfully: {archive_path} ({num_files} files, took {elapsed_time:.2f} seconds)")
    
    return num_files




def distribute_by_size(archives_with_sizes, world_size):
    """
    Distribute archives across processes so each process gets roughly equal total size.
    Uses greedy algorithm: assign each archive to the process with smallest current load.
    
    Args:
        archives_with_sizes: List of (archive_path, size_bytes)
        world_size: Number of processes
    
    Returns:
        List of lists, where result[pid] contains archives for that process
    """
    # Sort by size descending for better distribution
    sorted_archives = sorted(archives_with_sizes, key=lambda x: x[1], reverse=True)
    
    # Initialize buckets for each process
    buckets = [[] for _ in range(world_size)]
    bucket_sizes = [0] * world_size
    
    # Greedy assignment
    for archive, size in sorted_archives:
        # Find bucket with minimum size
        min_idx = bucket_sizes.index(min(bucket_sizes))
        buckets[min_idx].append((archive, size))
        bucket_sizes[min_idx] += size
    
    return buckets, bucket_sizes


def get_backed_up_folders(backup_file):
    """Get the set of relative paths that have already been backed up."""
    if not os.path.exists(backup_file):
        return set()

    backed_up = set()
    try:
        with open(backup_file, 'r') as f:
            backed_up = {line.strip() for line in f if line.strip()}
    except Exception as e:
        print(f"Warning: Could not read backup.txt: {e}")
    return backed_up

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Create tar archives of processed datasets')
    parser.add_argument('--pid', type=int, required=True, help='Process ID (0 to world_size-1)')
    parser.add_argument('--world', type=int, required=True, help='Total number of processes')
    parser.add_argument('--dryrun', action='store_true', help='Simulate the backup process without making changes')
    parser.add_argument('--extract', action='store_true', help='Extract backup tar archives back to the source folders')
    parser.add_argument('--num_workers', type=int, default=1, help='Number of parallel workers for tar/untar operations')
    # write to a sibling dir under TRG_STORE so a re-backup never overwrites the old archive set
    parser.add_argument('--dst_name', type=str, default='datasets_preprocess_backup', help='Backup dir name under $TRG_STORE')
    args = parser.parse_args()

    if args.pid < 0 or args.pid >= args.world:
        print(f"Error: pid must be between 0 and {args.world-1}")
        exit(1)

    if args.dryrun:
        print("Running in dryrun mode - no changes will be made")

    SCRATCH = os.environ["SCRATCH"]
    TRG_STORE = os.environ["TRG_STORE"]
    src_dir = os.path.join(SCRATCH, "occany_dataset")
    dst = os.path.join(TRG_STORE, args.dst_name)
    
    backup_folders = [
        "waymo_processed",
        "ddad_processed",
        "pandaset_processed",
        "vkitti_processed",
        "once_processed",
        "kitti_processed",
        "nuscenes_processed",
        "occ3d_nuscenes_processed"   # present on fsn1 but was never backed up
    ]

    # ------------------------------------------------------------
    # Extraction Mode
    # ------------------------------------------------------------
    if args.extract:
        # Gather all archive files with their sizes
        archives_with_sizes = []
        if os.path.isdir(dst):
            for dataset_name in os.listdir(dst):
                if dataset_name not in backup_folders:
                    continue
                dataset_path = os.path.join(dst, dataset_name)
                if not os.path.isdir(dataset_path):
                    continue
                for f in os.listdir(dataset_path):
                    if f.endswith('.tar') or f.endswith('.tar.gz'):
                        rel_path = os.path.join(dataset_name, f)
                        abs_path = os.path.join(dst, rel_path)
                        size = os.path.getsize(abs_path)
                        archives_with_sizes.append((rel_path, size))
        
        if len(archives_with_sizes) == 0:
            print("No archives found to extract")
            exit(0)

        # Distribute archives by size across processes
        buckets, bucket_sizes = distribute_by_size(archives_with_sizes, args.world)
        
        # Get this process's archives
        my_archives = buckets[args.pid]
        my_size = bucket_sizes[args.pid]
        total_size = sum(bucket_sizes)
        
        if len(my_archives) == 0:
            print(f"Process {args.pid} has no archives to extract")
            exit(0)
        
        print(f"Process {args.pid}/{args.world} will extract {len(my_archives)}/{len(archives_with_sizes)} archives ")
        print(f"  Size: {my_size / 1e9:.2f} GB / {total_size / 1e9:.2f} GB total ({100*my_size/total_size:.1f}%)")

        # Prepare extraction tasks
        extract_tasks = []
        for rel_path, _ in my_archives:
            dataset_name = os.path.dirname(rel_path)
            src_dataset_dir = os.path.join(src_dir, dataset_name)
            archive_abs_path = os.path.join(dst, rel_path)

            if not args.dryrun:
                os.makedirs(src_dataset_dir, exist_ok=True)

            extract_tasks.append((archive_abs_path, src_dataset_dir, args.dryrun))

        # Run extraction with local progress tracking
        total_local = len(extract_tasks)
        
        if args.num_workers > 1:
            with Pool(args.num_workers) as pool:
                for i, _ in enumerate(pool.starmap(extract_tar_archive, extract_tasks), 1):
                    print(f"[Process {args.pid}] {i}/{total_local} archives extracted")
        else:
            for i, task in enumerate(extract_tasks, 1):
                extract_tar_archive(*task)
                print(f"[Process {args.pid}] {i}/{total_local} archives extracted")

        # Extraction complete
        exit(0)

    # Define all possible source folders (relative paths)
    all_source_folders = []
    for folder in backup_folders:
        full_path = os.path.join(src_dir, folder)
        if os.path.isdir(full_path):
            subdirs = [os.path.join(folder, d) for d in os.listdir(full_path) if os.path.isdir(os.path.join(full_path, d)) and d != "tmp"]
            all_source_folders.extend(subdirs)
            
            # Copy all .npz files in each folder
            npz_files = [f for f in os.listdir(full_path) if f.endswith('.npz')]
            for npz_file in npz_files:
                src_file = os.path.join(full_path, npz_file)
                os.makedirs(os.path.join(dst, folder), exist_ok=True)
                dst_file = os.path.join(dst, folder, npz_file)
                if not args.dryrun:
                    shutil.copy2(src_file, dst_file)
                print(f"Copied {src_file} to {dst_file}")

    # Sort folders to ensure consistent distribution across processes
    all_source_folders.sort()

    # Get already backed up folders
    # backup_file = os.path.join(dst, "backup.txt")
    # backed_up_folders = get_backed_up_folders(backup_file)

   
    # Filter out already backed up folders
    # all_source_folders = [f for f in all_source_folders if f not in backed_up_folders]
    # if not all_source_folders:
    #     print("All folders have already been backed up")
    #     exit(0)



    # Calculate which folders this process should handle
    total_folders = len(all_source_folders)
    folders_per_process = (total_folders + args.world - 1) // args.world  # Ceiling division
    start_idx = args.pid * folders_per_process
    end_idx = min(start_idx + folders_per_process, total_folders)


    if start_idx >= total_folders:
        print(f"Process {args.pid} has no folders to process")
        exit(0)

    source_folders = all_source_folders[start_idx:end_idx]
    print(f"Process {args.pid}/{args.world} will process {len(source_folders)}/{total_folders} folders (indices {start_idx} to {end_idx-1})")

    if not args.dryrun:
        os.makedirs(dst, exist_ok=True)

    # Prepare tar tasks
    tar_tasks = []
    for folder in source_folders:
        # Get the parent dataset name and create corresponding directory in dst
        dataset_name = os.path.dirname(folder).split(os.sep)[0]  # Get first part of relative path
        dst_dataset_dir = os.path.join(dst, dataset_name)
        if not args.dryrun:
            os.makedirs(dst_dataset_dir, exist_ok=True)

        tar_tasks.append((folder, dst_dataset_dir, src_dir, args.dryrun))

    # Run tar with multiprocessing
    if args.num_workers > 1:
        with Pool(args.num_workers) as pool:
            list(tqdm(pool.starmap(create_tar_archive, tar_tasks), total=len(tar_tasks)))
    else:
        for task in tqdm(tar_tasks):
            print(f"Processing {task[0]} and saving to {task[1]}")
            create_tar_archive(*task)
        
        
