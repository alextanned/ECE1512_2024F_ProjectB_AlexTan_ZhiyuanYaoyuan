
## Part 1 & 2

The majority of the boiler plate code is from [HEDNet](https://github.com/zhanggang001/HEDNet)

### For Single Frame Reconstruction
```bash
python ./tools/train.py --cfg_file tools/cfgs/mamba_models/label_raster_fsq_train.yaml --output_dir outputs
```

#### For Multi Frame Reconstruction with Noise and with Mamba

```bash
# single gpu
python ./tools/train.py --cfg_file tools/cfgs/mamba-models/label_raster_seq2seq_train.yaml --output_dir outputs
```
