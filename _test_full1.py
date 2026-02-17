import time, sys, os
os.environ['PYTHONIOENCODING'] = 'utf-8'
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from main import process_panorama, get_default_config

config = get_default_config()
config['profile'] = True

t0 = time.time()
result = process_panorama('full1.jpg', 'output/full1', config)
elapsed = time.time() - t0

print(f'\nTotal: {elapsed:.1f}s')
print(f'Success: {result["success"]}')
