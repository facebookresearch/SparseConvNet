[ScanNet](http://www.scan-net.org/)
-------

To train a small U-Net with 5cm-cubed sparse voxels:

1. Download [ScanNet](http://www.scan-net.org/) files
2. [Split](https://github.com/ScanNet/ScanNet/tree/master/Tasks/Benchmark) the files *vh_clean_2.ply and *_vh_clean_2.labels.ply files into 'train/' and 'val/' folders
3. Run 'pip install plyfile'
4. Run 'python prepare_data.py'
5. Run 'python unet.py'

You can the computational cost (and hopefully accuracy too) by changing m / block_reps / residual_blocks / scale / val_reps in unet.py / data.py.
