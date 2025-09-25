import time
import os
import sqlite3
import threading
from concurrent.futures import ThreadPoolExecutor
from queue import Queue, Empty
from threading import Thread, Event
import pickle
class DirectoryIndexDict:
    def __init__(self, verbose=True, dict_path=None, exclude=['temp'], save_interval=60):
        self.verbose = verbose
        self.refresh_queue = Queue()
        self.queue_worker_running = False
        self.processing_paths = set()
        self.processed_paths = {}  # path -> last processed timestamp
        self.completion_events = {}  # path -> threading.Event
        self.db = {}  # parent_path -> list of children
        self.db_lock = threading.RLock()
        self.dict_path = dict_path
        self.exclude = exclude
        self.save_interval = save_interval
        self._start_queue_worker()
        if self.dict_path:
            self._start_periodic_save_thread()

    def _log(self, message):
        if self.verbose:
            print(message)

    def _normalize_path(self, path):
        if not path:
            return path
        normalized = os.path.abspath(path)
        if len(normalized) > 1 and normalized.endswith(os.sep):
            normalized = normalized.rstrip(os.sep)
        return normalized

    def _start_queue_worker(self):
        if not self.queue_worker_running:
            self.queue_worker_running = True
            worker_thread = Thread(target=self._queue_worker, daemon=True)
            worker_thread.start()
            self._log("Directory index queue worker started")

    def _queue_worker(self):
        self._log("Queue worker thread started")
        while self.queue_worker_running:
            try:
                task = self.refresh_queue.get(timeout=1)
                if task is None:
                    self._log("Queue worker received stop signal")
                    break
                parent_path = self._normalize_path(task)
                now = time.time()
                # Skip if already processing
                if parent_path in self.processing_paths:
                    self.refresh_queue.task_done()
                    continue
                # Skip if processed recently (within 30s)
                if parent_path in self.processed_paths and now - self.processed_paths[parent_path] < 30:
                    self.refresh_queue.task_done()
                    # Signal completion even if skipped
                    if parent_path in self.completion_events:
                        self.completion_events[parent_path].set()
                    continue
                self._log(f"Processing queued refresh for: {parent_path} (Queue size: {self.refresh_queue.qsize()})")
                self.processing_paths.add(parent_path)
                try:
                    success = self._refresh_directory_sync(parent_path)
                    if success:
                        self.processed_paths[parent_path] = now
                        self._log(f"Successfully processed: {parent_path}")
                    else:
                        self._log(f"Failed to process: {parent_path}")
                    # Signal completion regardless of success/failure
                    if parent_path in self.completion_events:
                        self.completion_events[parent_path].set()
                finally:
                    self.processing_paths.discard(parent_path)
                # Clean up old processed paths (keep only last hour)
                old_paths = [path for path, timestamp in self.processed_paths.items()
                             if now - timestamp > 3600]
                for path in old_paths:
                    del self.processed_paths[path]
                self.refresh_queue.task_done()
            except Empty:
                continue
            except Exception as e:
                import traceback
                self._log(f"Queue worker error: {type(e).__name__}: {str(e)}")
                self._log(f"Traceback: {traceback.format_exc()}")
                # Signal completion even on error
                if 'parent_path' in locals() and parent_path in self.completion_events:
                    self.completion_events[parent_path].set()
                try:
                    self.refresh_queue.task_done()
                except Exception:
                    pass
        self._log("Queue worker thread stopped")

    def _start_periodic_save_thread(self):
        def periodic_save():
            while self.queue_worker_running:
                time.sleep(self.save_interval)
                self.save_dict(self.dict_path)
        thread = threading.Thread(target=periodic_save, daemon=True)
        thread.start()
        self._log(f"Started periodic save thread every {self.save_interval} seconds")

    def get_queue_size(self):
        return self.refresh_queue.qsize()

    def get_processing_status(self):
        return {
            'queue_size': self.refresh_queue.qsize(),
            'processing_paths': list(self.processing_paths),
            'processed_count': len(self.processed_paths)
        }

    def clear_queue(self):
        while not self.refresh_queue.empty():
            try:
                self.refresh_queue.get_nowait()
                self.refresh_queue.task_done()
            except:
                break
        self.processed_paths.clear()
        self.processing_paths.clear()
        for event in self.completion_events.values():
            event.set()
        self.completion_events.clear()
        self._log("Queue cleared and processed paths reset")

    def close(self):
        self._log("Shutting down DirectoryIndexDict...")
        self.queue_worker_running = False
        self.refresh_queue.put(None)
        for event in self.completion_events.values():
            event.set()
        try:
            self.refresh_queue.join()
        except Exception as e:
            self._log(f"Error joining queue: {e}")
        self._log("DirectoryIndexDict shutdown complete")

    def get_children(self, parent_path, directories_only=True, wait_if_needed=True, timeout=300):
        """
        Get children from dict with flexible refresh and waiting options.
        """
        parent_path = self._normalize_path(parent_path)
        with self.db_lock:
            items = self.db.get(parent_path, [])
            # Only use dict entries, skip strings
            items = [p for p in items if isinstance(p, dict) and 'path' in p and 'is_directory' in p]
            if directories_only:
                children = [os.path.basename(p['path']) for p in items if p['is_directory']]
            else:
                children = [{'name': os.path.basename(p['path']), 'is_directory': p['is_directory']} for p in items]

        # If wait_if_needed and no children but path exists, queue and wait
        if wait_if_needed and not children and os.path.exists(parent_path):
            self._log(f"No data found for existing path {parent_path}, queuing and waiting...")
            completion_event = Event()
            self.completion_events[parent_path] = completion_event
            try:
                if parent_path not in self.processing_paths:
                    self.refresh_queue.put(parent_path)
                if completion_event.wait(timeout=timeout):
                    self._log(f"Queue processing completed for {parent_path}, retrying query...")
                    with self.db_lock:
                        items = self.db.get(parent_path, [])
                        items = [p for p in items if isinstance(p, dict) and 'path' in p and 'is_directory' in p]
                        if directories_only:
                            children = [os.path.basename(p['path']) for p in items if p['is_directory']]
                        else:
                            children = [{'name': os.path.basename(p['path']), 'is_directory': p['is_directory']} for p in items]
                else:
                    self._log(f"Timeout waiting for queue processing of {parent_path}")
            finally:
                self.completion_events.pop(parent_path, None)
        else:
            should_refresh = False
            if not children:
                should_refresh = True
                reason = "no data found"
            else:
                now = time.time()
                if parent_path not in self.processed_paths:
                    should_refresh = True
                    reason = "initial refreshing"
                elif now - self.processed_paths[parent_path] > 300:
                    should_refresh = True
                    reason = "data is old"
            if should_refresh and parent_path not in self.processing_paths:
                self._log(f"Queuing refresh for {parent_path} ({reason})")
                self.refresh_queue.put(parent_path)
        return children

    def _refresh_directory_sync(self, parent_path):
        parent_path = self._normalize_path(parent_path)
        try:
            if not os.path.exists(parent_path):
                self._log(f"Path does not exist: {parent_path}")
                return False
            if not os.path.isdir(parent_path):
                self._log(f"Path is not directory: {parent_path}")
                return False
            # Get current directory contents (directories only, fast)
            current_items = []
            with os.scandir(parent_path) as it:
                for entry in it:
                    if not entry.name.startswith('.') and entry.is_dir() and not entry in self.exclude:
                        current_items.append({'path': entry.path, 'is_directory': True})
            with self.db_lock:
                self.db[parent_path] = current_items
            self._log(f"Background refresh: updated {len(current_items)} items for {parent_path}")
            if self.dict_path:
                self.save_dict(self.dict_path)
            return True
        except Exception as e:
            self._log(f"Error in background refresh for {parent_path}: {e}")
            return False
        
    def push(self, path_list):
        if not isinstance(path_list, list):
            path_list = [path_list]

        for path in path_list:
            need_to_push = True
            basename = os.path.basename(path)
            parent_name = os.path.dirname(path)
            if parent_name in self.db:
                dir_list = [p['path'] for p in self.db[parent_name]]
                if basename in dir_list:
                    need_to_push = False # Already exists
                else:
                    pass

            if need_to_push:
                # Append if needed
                self.db[parent_name].append({'path': path, 'is_directory': os.path.isdir(path)})
                
    def save_dict(self, filename):
        with self.db_lock:
            with open(filename, "wb") as f:
                pickle.dump(self.db, f)
        self._log(f"Saved directory index to {filename}")

    def load_dict(self, filename):
        if os.path.exists(filename):
            with self.db_lock:
                with open(filename, "rb") as f:
                    self.db = pickle.load(f)
            self._log(f"Loaded directory index from {filename}")
        else:
            self._log(f"File {filename} does not exist. Starting with empty index.")

    def set_verbose(self, verbose):
        self.verbose = verbose

_dir_cache = {}
_cache_ttl = 60

def _scan_directory(dir_path, include_files=False, return_type_info=False):
    """Scan directory and return items"""
    items = []
    try:
        for item in os.listdir(dir_path):
            if item.startswith('.'):
                continue
                
            item_path = os.path.join(dir_path, item)
            is_dir = os.path.isdir(item_path)
            is_file = os.path.isfile(item_path)
            
            if is_dir or (include_files and is_file):
                if return_type_info:
                    items.append({
                        'name': item,
                        'is_directory': is_dir,
                        'is_file': is_file
                    })
                else:
                    items.append(item)
    except Exception as e:
        print(f"Error scanning directory {dir_path}: {e}")
    
    return items

def get_cached_dirs(dir_path, include_files=False, return_type_info=False):
    """Backward compatible cached directory listing"""
    now = time.time()
    cache_key = f"{dir_path}_{include_files}_{return_type_info}"
    
    if cache_key in _dir_cache:
        data, timestamp = _dir_cache[cache_key]
        if now - timestamp < _cache_ttl:
            return data
    
    if os.path.exists(dir_path):
        try:
            items = _scan_directory(dir_path, include_files, return_type_info)
            items.sort(key=lambda x: x['name'] if isinstance(x, dict) else x)
            _dir_cache[cache_key] = (items, now)
            return items
        except Exception as e:
            print(f"Error reading directory {dir_path}: {e}")
    
    return []


if __name__ == "__main__":
    data_root_dir = "/home/heesup/GEMINI-App-Data"
    db_path = os.path.join(data_root_dir, "directory_index.db")
    dict_path = os.path.join(data_root_dir, "directory_index_dict.pkl")

    test_paths = [
        os.path.join(data_root_dir, 'Raw'),
        os.path.join(data_root_dir, 'Processed'),
        data_root_dir
    ]


    # --- Test Dictionary-based DirectoryIndexDict ---
    print("=== DirectoryIndexDict (Dict) Timing Check ===")
    t2 = time.time()
    dir_dict = DirectoryIndexDict(verbose=False)
    # Try loading from file if exists
    if os.path.exists(dict_path):
        dir_dict.load_dict(dict_path)
        print(f"Loaded dict from {dict_path}")
    else:
        print(f"No dict file found, will build index from scratch.")

    for test_path in test_paths:
        if os.path.exists(test_path):
            try:
                children = dir_dict.get_children(test_path, directories_only=True, wait_if_needed=True)
                print(f"Path: {test_path}")
                print(f"  Exists: True")
                print(f"  Children count: {len(children)}")
                print(f"  Children: {children[:5] if len(children) > 5 else children}")
            except Exception as e:
                print(f"Path: {test_path}")
                print(f"  Error getting children: {e}")
        else:
            print(f"Path: {test_path}")
            print(f"  Exists: False")
    # Save dict for future runs
    dir_dict.save_dict(dict_path)
    t3 = time.time()
    print(f"Dict DirectoryIndex elapsed: {t3-t2:.4f} seconds")
    status = dir_dict.get_processing_status()
    print(f"Processing status: {status}")
    print("=== End DirectoryIndexDict (Dict) Check ===")