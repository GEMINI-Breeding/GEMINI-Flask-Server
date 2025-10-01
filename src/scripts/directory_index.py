import time
import os
import sqlite3
import threading
from concurrent.futures import ThreadPoolExecutor
from queue import Queue, Empty
from threading import Thread, Event
import pickle
import stat

from utils import normalize_path
class DirectoryIndexDict:
    def __init__(self, dict_path, verbose=True, exclude=['temp'], save_interval=60):
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
                parent_path = normalize_path(task)
                now = time.time()
                # Skip if already processing
                if parent_path in self.processing_paths:
                    self.refresh_queue.task_done()
                    continue
                # Skip if processed recently (within 300s)
                if parent_path in self.processed_paths and now - self.processed_paths[parent_path] < 300:
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

    def get_children(self, parent_path, directories_only=True, wait_if_needed=False, timeout=300):
        """
        Get children immediately from dict and queue refresh if needed.
        Returns tuple (children, is_fresh) where is_fresh indicates if data was recently updated.
        """
        parent_path = normalize_path(parent_path)
        is_fresh = False

        # 1. Immediately return current data from dictionary
        with self.db_lock:
            items = self.db.get(parent_path, [])
            items = [p for p in items if isinstance(p, dict) and 'path' in p and 'is_directory' in p]
            if directories_only:
                children = [os.path.basename(p['path']) for p in items if p['is_directory']]
            else:
                children = [{'name': os.path.basename(p['path']), 'is_directory': p['is_directory']} for p in items]

        # 2. Check if refresh is needed
        now = time.time()
        needs_refresh = False

        # Check current state
        is_being_processed = parent_path in self.processing_paths
        is_in_queue = any(task == parent_path for task in list(self.refresh_queue.queue))
        is_stale = parent_path not in self.processed_paths or now - self.processed_paths[parent_path] > 300

        # Determine if refresh is needed
        if not children and os.path.exists(parent_path):
            needs_refresh = not (is_being_processed or is_in_queue)
        elif is_stale:
            needs_refresh = not (is_being_processed or is_in_queue)

        # 3. Queue refresh if needed
        if needs_refresh:
            self._log(f"Queuing background refresh for {parent_path} (In process: {is_being_processed}, In queue: {is_in_queue})")
            self.refresh_queue.put(parent_path)
            
            # Only wait if explicitly requested
            if wait_if_needed:
                completion_event = Event()
                self.completion_events[parent_path] = completion_event
                try:
                    if completion_event.wait(timeout=timeout):
                        # Refresh complete, get updated data
                        with self.db_lock:
                            items = self.db.get(parent_path, [])
                            items = [p for p in items if isinstance(p, dict) and 'path' in p and 'is_directory' in p]
                            if directories_only:
                                children = [os.path.basename(p['path']) for p in items if p['is_directory']]
                            else:
                                children = [{'name': os.path.basename(p['path']), 'is_directory': p['is_directory']} for p in items]
                        is_fresh = True
                finally:
                    self.completion_events.pop(parent_path, None)
        else:
            is_fresh = parent_path in self.processed_paths and now - self.processed_paths[parent_path] < 300

        return children

    def _refresh_directory_sync(self, parent_path):
        parent_path = normalize_path(parent_path)
        try:
            # Add quick path checks
            if not os.access(parent_path, os.R_OK):
                self._log(f"No read access: {parent_path}")
                return False
                
            # Use stat to get directory info quickly
            st = os.stat(parent_path)
            if not stat.S_ISDIR(st.st_mode):
                return False

            # Fast directory scan with list comprehension
            with os.scandir(parent_path) as it:
                current_items = [
                    {'path': entry.path, 'is_directory': True}
                    for entry in it 
                    if entry.is_dir(follow_symlinks=False) and  # Don't follow symlinks
                       not entry.name.startswith('.') and 
                       entry.name not in self.exclude
                ]

            with self.db_lock:
                self.db[parent_path] = current_items
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