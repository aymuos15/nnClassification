# Best Practices

Tips and conventions for effective use of the framework.

## Configuration

1. **Start with defaults** - Modify incrementally
2. **Version control configs** - Track experiment settings
3. **Document changes** - Note why you changed defaults
4. **Use meaningful names** - For custom configs
5. **Check saved config** - Verify what was actually used

## Training

1. **Test quickly first** - Few epochs to verify pipeline
2. **Monitor with TensorBoard** - Watch training live
3. **Check GPU utilization** - Should be near 100%
4. **Save checkpoints** - Always use checkpointing
5. **Track experiments** - Document what works

## Hyperparameter Tuning

1. **One at a time** - Isolate effects
2. **Start coarse** - Wide range, then narrow
3. **Use TensorBoard** - Visual comparison
4. **Be systematic** - Grid or random search
5. **Document results** - Track all experiments

## Data

1. **Verify structure** - Use verification script
2. **Check class balance** - Ensure fair distribution
3. **Use validation set** - Don't overfit to test
4. **Augment appropriately** - Match your domain
5. **Inspect samples** - Verify preprocessing

## Code

1. **Don't modify entry points** - Extend via modules
2. **Test changes** - Before full training
3. **Follow conventions** - Match existing patterns
4. **Document extensions** - Help future users
5. **Keep backups** - Before major changes

## Reproducibility

1. **Set seed** - Always use fixed seed
2. **Save config** - With each run
3. **Document environment** - Python/PyTorch versions
4. **Use deterministic** - When exact reproduction needed
5. **Track hardware** - GPU model affects results

## Performance

1. **Maximize batch size** - Within memory limits
2. **Use appropriate workers** - 4-8 typically good
3. **Non-deterministic default** - Faster training
4. **Monitor bottlenecks** - CPU, GPU, or I/O
5. **Profile if needed** - Find slow components

## Deployment

1. **Use best.pt** - Highest validation accuracy
2. **Test inference** - Before deployment
3. **Document model** - Architecture and training details
4. **Save transforms** - Needed for inference
5. **Version models** - Track which version deployed

## Related

- [Training Guide](../user-guides/training.md)
- [Configuration](../configuration/README.md)
- [Troubleshooting](troubleshooting.md)
