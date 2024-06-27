# [Git](https://github.com/cuda-mode/lectures/tree/main/lecture_001)

# [slides](https://docs.google.com/presentation/d/110dnMW94LX1ySWxu9La17AVUxjgSaQDLOotFC3BZZD4/edit#slide=id.g2658e4ac9dd_0_0)

# Goal

1. Integrate a CUDA kernel inside PyTorch progra,
2. Learn how to profile it 

- CUDA is ASYNC so you can't use the time's python library. You'll compute only the overhead it takes to launch a kernel

```python
def time_pytorch_function(func, input):
    # CUDA IS ASYNC so can't use python time module
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    # Warmup because the 1st time you call CUDA in a PyTorch function it's gonna initialize the CUDA context
     
    for _ in range(5):
        func(input)

    start.record()
    func(input)
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end)
```

