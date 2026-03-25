recommend-cold-items=TRUE, filter-cold-items=FALSE：
最完整冷启动设置。冷物品既能出现在用户历史里，也能被推荐出来。

recommend-cold-items=TRUE, filter-cold-items=TRUE：
冷物品只能作为候选被推荐，不能出现在输入历史里。

recommend-cold-items=FALSE, filter-cold-items=FALSE：
冷物品可以出现在历史里，但不能被推荐。

recommend-cold-items=FALSE, filter-cold-items=TRUE：
冷物品既不进历史，也不进候选，最接近“普通 SASRec 不处理冷物品”

解释来源：
这两个参数都只影响评测阶段，不影响训练。

先看代码里的真实含义。

recommend-cold-items
在 lightning.py 里控制候选集合：

True：cold item 也允许进入候选集并被推荐
False：只在 [: num_items + 1] 这个 warm item 范围里排序，cold item 不允许被推荐
也就是说，它回答的是：

预测时，模型能不能把 cold item 排进 Top-K？

filter-cold-items
在 pipeline.py 里控制输入历史：

True：先把 interactions 里 is_cold=True 的交互过滤掉，再送进测试 dataloader
False：测试用户历史保留原样，冷物品也可以出现在输入序列里
也就是说，它回答的是：

测试时，用户历史里允不允许出现 cold item？


所以，真正重点看 recommend-cold-items = TRUE 的两组。
另外，recommend-cold-items = FALSE 的两组主要用来看：你的方法会不会伤害 warm/overall。