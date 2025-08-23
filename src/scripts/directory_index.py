import time
import os
import sqlite3
import threading
from concurrent.futures import ThreadPoolExecutor
from queue import Queue, Empty
from threading import Thread, Event

class DirectoryIndex:
    def __init__(self, db_path="directory_index.db", verbose=True):
        self.db_path = db_path
        self.verbose = verbose  # Add verbose class variable
        self.lock = threading.RLock()
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="dir_index")
        self.refresh_queue = Queue()
        self.queue_worker_running = False
        self.processing_paths = set()  # Paths currently being processed
        self.processed_paths = {}  # Paths with their last processed timestamp
        self.completion_events = {}  # Events to signal when specific paths are processed
        self._init_db()
        self._start_queue_worker()
    
    def _log(self, message):
        """Print message only if verbose is enabled"""
        if self.verbose:
            print(message)
    
    def _normalize_path(self, path):
        """Normalize path by removing trailing slash and converting to absolute path"""
        if not path:
            return path
        
        # Convert to absolute path and normalize
        normalized = os.path.abspath(path)
        
        # Remove trailing slash except for root directory
        if len(normalized) > 1 and normalized.endswith(os.sep):
            normalized = normalized.rstrip(os.sep)
            
        return normalized
    
    def _init_db(self):
        """Initialize simple database structure"""
        try:
            with sqlite3.connect(self.db_path, timeout=30) as conn:
                conn.execute('PRAGMA journal_mode=WAL')
                conn.execute('PRAGMA synchronous=NORMAL')
                conn.execute('PRAGMA cache_size=10000')
                
                # Check if table exists and get record count
                cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='files'")
                table_exists = cursor.fetchone() is not None
            
                conn.execute('''
                    CREATE TABLE IF NOT EXISTS files (
                        path TEXT PRIMARY KEY,
                        parent TEXT NOT NULL,
                        name TEXT NOT NULL,
                        is_directory INTEGER NOT NULL,
                        last_modified REAL
                    )
                ''')
                conn.execute('CREATE INDEX IF NOT EXISTS idx_parent ON files(parent)')
                conn.commit()
            
                # Get total record count
                cursor = conn.execute('SELECT COUNT(*) FROM files')
                total_records = cursor.fetchone()[0]
            
                # Get sample records
                cursor = conn.execute('SELECT parent, name, is_directory FROM files LIMIT 5')
                sample_records = cursor.fetchall()
            
                self._log("Database initialized successfully")
                self._log(f"  Table existed: {table_exists}")
                self._log(f"  Total records: {total_records}")
                self._log(f"  Sample records: {sample_records}")
            
        except Exception as e:
            self._log(f"Error initializing database: {e}")
    
    def _start_queue_worker(self):
        """Start background queue worker"""
        if not self.queue_worker_running:
            self.queue_worker_running = True
            worker_thread = Thread(target=self._queue_worker, daemon=True)
            worker_thread.start()
            self._log("Directory index queue worker started")
    
    def _queue_worker(self):
        """Background worker that processes refresh tasks sequentially"""
        self._log("Queue worker thread started")
        while self.queue_worker_running:
            try:
                task = self.refresh_queue.get(timeout=1)
                if task is None:  # Poison pill to stop worker
                    self._log("Queue worker received stop signal")
                    break
                
                # Normalize the path
                parent_path = self._normalize_path(task)
                
                # Skip if currently being processed
                if parent_path in self.processing_paths:
                    self.refresh_queue.task_done()
                    continue
                
                # Skip if processed recently (within last 30 seconds)
                now = time.time()
                if parent_path in self.processed_paths:
                    if now - self.processed_paths[parent_path] < 30:
                        self.refresh_queue.task_done()
                        # Signal completion even if skipped
                        if parent_path in self.completion_events:
                            self.completion_events[parent_path].set()
                        continue
            
                # Print queue size when processing queued refresh
                queue_size = self.refresh_queue.qsize()
                self._log(f"Processing queued refresh for: {parent_path} (Queue size: {queue_size})")
                
                # Mark as currently processing
                self.processing_paths.add(parent_path)
                
                try:
                    success = self._refresh_directory_sync(parent_path)
                    if success:
                        # Mark as successfully processed
                        self.processed_paths[parent_path] = now
                        self._log(f"Successfully processed: {parent_path}")
                    else:
                        self._log(f"Failed to process: {parent_path}")
                        
                    # Signal completion regardless of success/failure
                    if parent_path in self.completion_events:
                        self.completion_events[parent_path].set()
                        
                finally:
                    # Remove from processing set
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
                except ValueError:
                    pass
    
        self._log("Queue worker thread stopped")
    
    def _get_db_connection(self, timeout=10):
        """Get database connection with timeout and WAL mode"""
        conn = sqlite3.connect(self.db_path, timeout=timeout)
        conn.execute('PRAGMA journal_mode=WAL')
        return conn
    
    def _insert_file_record(self, conn, path, parent=None):
        """Insert a single file record into database"""
        # Normalize paths before storing
        path = self._normalize_path(path)
        parent = self._normalize_path(parent) if parent else self._normalize_path(os.path.dirname(path))
        
        if not os.path.exists(path):
            return False
            
        try:
            name = os.path.basename(path)
            is_directory = 1 if os.path.isdir(path) else 0
            last_modified = os.path.getmtime(path)
            
            conn.execute('''
                INSERT OR REPLACE INTO files 
                (path, parent, name, is_directory, last_modified) 
                VALUES (?, ?, ?, ?, ?)
            ''', (path, parent, name, is_directory, last_modified))
            return True
        except Exception as e:
            self._log(f"Error inserting file record for {path}: {e}")
            return False
    
    def _query_children(self, parent_path, directories_only=True):
        """Query children from database with connection timeout"""
        # Normalize parent_path before querying
        parent_path = self._normalize_path(parent_path)
        
        children = []
        try:
            with self._get_db_connection(timeout=5) as conn:
                if directories_only:
                    cursor = conn.execute(
                        'SELECT name FROM files WHERE parent = ? AND is_directory = 1 ORDER BY name',
                        (parent_path,)
                    )
                    result = cursor.fetchall()
                    children = [row[0] for row in result]
                else:
                    cursor = conn.execute(
                        'SELECT name, is_directory FROM files WHERE parent = ? ORDER BY name',
                        (parent_path,)
                    )
                    result = cursor.fetchall()
                    children = [{'name': row[0], 'is_directory': bool(row[1])} for row in result]
                    
        except sqlite3.OperationalError as e:
            if "database is locked" in str(e):
                self._log(f"Database locked for {parent_path}, returning empty result")
                return []
            else:
                self._log(f"Database error for {parent_path}: {e}")
        except Exception as e:
            self._log(f"Query error for {parent_path}: {e}")
        
        return children
    
    def update_paths(self, paths):
        """Update database with list of file/directory paths"""
        if not paths:
            return
            
        with self.lock:
            try:
                with self._get_db_connection(timeout=15) as conn:
                    success_count = 0
                    for path in paths:
                        if self._insert_file_record(conn, path):
                            success_count += 1
                    
                    conn.commit()
                    self._log(f"Updated {success_count}/{len(paths)} entries in database")
                    
            except Exception as e:
                self._log(f"Error updating database: {e}")
    
    def _refresh_directory_sync(self, parent_path):
        """Synchronous directory refresh (called by queue worker)"""
        # Normalize path
        parent_path = self._normalize_path(parent_path)
        
        try:
            if not os.path.exists(parent_path):
                self._log(f"Path does not exist: {parent_path}")
                return False
                
            if not os.path.isdir(parent_path):
                self._log(f"Path is not directory: {parent_path}")
                return False
                
            # Get current directory contents
            current_items = []
            try:
                for item in os.listdir(parent_path):
                    if not item.startswith('.'):  # Skip hidden files
                        item_path = os.path.join(parent_path, item)
                        current_items.append(item_path)
            except PermissionError:
                self._log(f"Permission denied accessing: {parent_path}")
                return False
            except Exception as e:
                self._log(f"Error reading directory {parent_path}: {e}")
                return False
            
            # Update database
            with self.lock:
                try:
                    with self._get_db_connection(timeout=20) as conn:
                        # Remove all existing entries for this parent
                        conn.execute('DELETE FROM files WHERE parent = ?', (parent_path,))
                        
                        # Insert current directory contents
                        success_count = 0
                        for item_path in current_items:
                            if self._insert_file_record(conn, item_path, parent_path):
                                success_count += 1
                        
                        conn.commit()
                        self._log(f"Background refresh: updated {success_count}/{len(current_items)} items for {parent_path}")
                        
                        # Verify the data was saved
                        verification = conn.execute(
                            'SELECT COUNT(*) FROM files WHERE parent = ?', 
                            (parent_path,)
                        ).fetchone()[0]
                        
                        if verification > 0:
                            self._log(f"Verification: {verification} entries saved for {parent_path}")
                            return True
                        else:
                            self._log(f"Warning: No entries found after saving for {parent_path}")
                            return False
                            
                except sqlite3.OperationalError as e:
                    if "database is locked" in str(e):
                        self._log(f"Database locked during refresh of {parent_path}")
                        return False
                    else:
                        self._log(f"Database error during refresh of {parent_path}: {e}")
                        return False
                        
        except Exception as e:
            self._log(f"Error in background refresh for {parent_path}: {e}")
            return False
    
    def get_children(self, parent_path, directories_only=True, wait_if_needed=False, timeout=300):
        """
        Get children from database with flexible refresh and waiting options
        
        Args:
            parent_path: Directory path to get children from
            directories_only: If True, return only directories
            wait_if_needed: If True, wait for queue processing when no data found but path exists
            timeout: Maximum seconds to wait for queue processing (only used when wait_if_needed=True)
        
        Returns:
            List of child names (directories_only=True) or list of dicts with name and is_directory
        """        
        # First: Quick database lookup
        children = self._query_children(parent_path, directories_only)
        
        # If wait_if_needed is True and no children found but path exists, wait for processing
        if wait_if_needed and not children and os.path.exists(parent_path):
            self._log(f"No data found for existing path {parent_path}, queuing and waiting...")
            
            # Create completion event for this path
            completion_event = Event()
            self.completion_events[parent_path] = completion_event
            
            try:
                # Queue the refresh if not already processing
                if parent_path not in self.processing_paths:
                    self.refresh_queue.put(parent_path)
                
                # Wait for processing to complete
                if completion_event.wait(timeout=timeout):
                    self._log(f"Queue processing completed for {parent_path}, retrying query...")
                    # Re-query the database after processing is complete
                    children = self._query_children(parent_path, directories_only)
                else:
                    self._log(f"Timeout waiting for queue processing of {parent_path}")
                    
            finally:
                # Clean up the completion event
                self.completion_events.pop(parent_path, None)
        
        # Handle async refresh logic (only if not already handled by wait_if_needed)
        else:
            should_refresh = False
            
            if not children:
                # No data found - need to refresh
                should_refresh = True
                reason = "no data found"
            else:
                # Data found - check if we should refresh in background
                now = time.time()
                if parent_path not in self.processed_paths:
                    should_refresh = True
                    reason = "initial refreshing"
                elif now - self.processed_paths[parent_path] > 300:  # 5 minutes
                    should_refresh = True
                    reason = "data is old"
            
            if should_refresh and parent_path not in self.processing_paths:
                self._log(f"Queuing refresh for {parent_path} ({reason})")
                self.refresh_queue.put(parent_path)
        
        return children
    
    def force_refresh(self, parent_path):
        """Synchronously refresh a directory"""
        # Normalize path
        parent_path = self._normalize_path(parent_path)
        
        # Remove from processed paths to force refresh
        self.processed_paths.pop(parent_path, None)
        self.processing_paths.discard(parent_path)
        
        success = self._refresh_directory_sync(parent_path)
        if success:
            self.processed_paths[parent_path] = time.time()
        
        return self._query_children(parent_path)
    
    def get_queue_size(self):
        """Get current queue size for monitoring"""
        return self.refresh_queue.qsize()
    
    def get_processing_status(self):
        """Get current processing status"""
        return {
            'queue_size': self.refresh_queue.qsize(),
            'processing_paths': list(self.processing_paths),
            'processed_count': len(self.processed_paths)
        }
    
    def clear_queue(self):
        """Clear the refresh queue"""
        while not self.refresh_queue.empty():
            try:
                self.refresh_queue.get_nowait()
                self.refresh_queue.task_done()
            except:
                break
        
        self.processed_paths.clear()
        self.processing_paths.clear()
        
        # Clear completion events
        for event in self.completion_events.values():
            event.set()
        self.completion_events.clear()
        self._log("Queue cleared and processed paths reset")
    
    def close(self):
        """Clean shutdown"""
        self._log("Shutting down DirectoryIndex...")
        
        # Stop queue worker
        self.queue_worker_running = False
        self.refresh_queue.put(None)  # Poison pill
        
        # Signal all pending completion events
        for event in self.completion_events.values():
            event.set()
        
        # Wait for queue to empty
        try:
            self.refresh_queue.join()
        except Exception as e:
            self._log(f"Error joining queue: {e}")
        
        # Shutdown executor
        self.executor.shutdown(wait=True)
        self._log("DirectoryIndex shutdown complete")

    def set_verbose(self, verbose):
        """Enable or disable verbose logging"""
        self.verbose = verbose

# Rest of the file remains the same...
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