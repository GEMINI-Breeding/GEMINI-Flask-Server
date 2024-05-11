import asyncio
from concurrent.futures import ThreadPoolExecutor


def build_nested_structure_sync(path, current_depth=0, max_depth=2):
    if current_depth >= max_depth:
        return {}
    
    structure = {}
    for child in path.iterdir():
        if child.is_dir():
            structure[child.name] = build_nested_structure_sync(child, current_depth+1, max_depth)
    return structure

async def build_nested_structure(path, current_depth=0, max_depth=2):
    loop = asyncio.get_running_loop()
    with ThreadPoolExecutor() as pool:
        return await loop.run_in_executor(pool, build_nested_structure_sync, path, current_depth, max_depth)

async def process_directories_in_parallel(base_dir, max_depth=2):
    directories = [d for d in base_dir.iterdir() if d.is_dir()]
    tasks = [build_nested_structure(d, 0, max_depth) for d in directories]
    nested_structures = await asyncio.gather(*tasks)
    
    combined_structure = {}
    for d, structure in zip(directories, nested_structures):
        combined_structure[d.name] = structure
    
    return combined_structure