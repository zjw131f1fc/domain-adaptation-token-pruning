1000样本
72.97/81.4(500step, warmup刚结束, 21.72%keep ratio，无adapter)
74.57/81.40(500step, 无warm up, 25.12 keep ratio)

没观察到warm up对acc有显著影响。现在的问题是如何阻止模式崩溃？

开始测试，baseline:
500step: 68/76.63
1000step: 42 出现模式崩溃
MSE：
500step: 68.92/76.63
1000step: 68.5
1500step 68.82

todo：gan的输入也许不应该做归一化？看一下attention聚合的结果进入FFN前有没有归一化，采用一样的归一化方法