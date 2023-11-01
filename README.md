# LLM-Quantization
I want to combine RPTQ and OmniQuant, then use mix precision quantization.

1. 将OmniQuant对激活值的量化从per token, dynamic 改成per_tensor, static.
2. 将OmniQuant对激活值的量化改成per_cluster, static. 这里还没有做reorder，只是把一个激活值张量分成了若干个类，每个类的channel数都是一样的。
3. 在OmniQuant之前加入reorder操作。reorder部分完全参考RPTQ。
4. 对权重的量化器由原来的均匀量化改成指数（对数）量化。
5. 对权重的量化引入混合bit成分，对每个output channel都有一个可学的bit位。
