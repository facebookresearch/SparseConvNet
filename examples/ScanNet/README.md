[ScanNet](http://www.scan-net.org/)
-------

To train a small U-Net with 5cm-cubed sparse voxels:

1. Download [ScanNet](http://www.scan-net.org/) files
2. [Split](https://github.com/ScanNet/ScanNet/tree/master/Tasks/Benchmark) the files *vh_clean_2.ply and *_vh_clean_2.labels.ply files into 'train/' and 'val/' folders
3. Run 'pip install plyfile'
4. Run 'python prepare_data.py'
5. Run 'python unet.py'

You can train a bigger/more accurate network by changing `m` / `block_reps` / `residual_blocks` / `scale` / `val_reps` in unet.py / data.py, e.g.
```
m=32 # Wider network
block_reps=2 # Deeper network
residual_blocks=True # ResNet style basic blocks
scale=50 # 1/50 m = 2cm voxels
val_reps=3 # Multiple views at test time
batch_size=5 # Fit in 16GB of GPU memory
```
