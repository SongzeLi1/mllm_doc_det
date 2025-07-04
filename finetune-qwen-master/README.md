# 文件入口
sft_llm.py

# 模型入口
finetune/models/qwen_forensics_model.py

# 模型架构
图片 -> LLM输出位置信息 -> SAM模型定位 -> 得出最终结果

# 脚本文件
bash /data1/yuyangxin/finetune-qwen/script/sft_llm.sh



阶段	可调整参数范围	训练目标设计
​第一阶段	冻结LLM, 训练新的分类头+新嵌入层	学习伪造区域定位特征
​第二阶段	微调LLM	使用混合数据强化跨模态关联
​第三阶段	对抗训练 指令数据增强推理能力

# 一、预训练数据构造
1. 多模态混合数据：收集高分辨率的原始图像（如自然场景、证件、文档等）及其元数据（EXIF信息、拍摄设备指纹等），同时引入文本描述
2. 篡改数据合成：
    - 传统篡改手段：复制-粘贴、拼接、擦除等操作，记录操作日志作为标签
    - 生成式篡改：利用扩散模型生成逼真的篡改区域，并标注篡改边界
    - 噪声注入：添加JPEG压缩伪影、重采样痕迹、对比度异常等扰动

# 二、任务设置
遮蔽图像局部区块：逐步遮蔽图像伪造区域和非伪造区域
双通道损失设计：定位损失（Dice Loss + Boundary-aware Loss） + 分类损失（Focal Loss） + 文本输出损失
​篡改一致性对比学习：使用InfoNCE损失拉近正样本在特征空间的距离，推远篡改样本


# 三、预训练
`
{
  "instruction": "分析该证件照是否存在PS痕迹，重点检查发际线边缘和瞳孔反光",
  "input": "<图像BASE64编码>",
  "output": "1. 发际线边缘检测到0.2px的羽化异常\n2. 瞳孔反光方向与光源位置矛盾"
}
`

# 四、数据集构建
{
    "conversations": [
    {"role": "user", "content": "<image> 这张证件照是否存在PS痕迹？"},
    {"role": "assistant", "content": 
      伪造区域: <mask_1> - <mask_32>；
      取证判断依据: 发际线羽化异常；
      取证难易判断: XXX;
      真实性结论：<cls_1> - <cls_32>";
    }
  ],
}

# 文本详细
`
{"messages": [{"role": "assistant", "content": "<image>是一张真实图像，<image>是XXX类型的伪造图像"}], "images": ["/xxx/x.jpg", "/xxx/x.png"]}
{"messages": [{"role": "user", "content": "<image><image>两张图片有什么区别"}, {"role": "assistant", "content": "前一张是真实图像, 后一张图像是伪造图像"}], "images": ["/xxx/x.jpg", "/xxx/x.png"]}
`

# 微调过程
基础模型: 实现Qwen2.5vl
过程
  - tokenizer添加特殊标记
    - <mask_1> - <mask_32>，用于mask训练
    - <cls>，用于分类任务
  - 实现微调模型
    - 提取<mask_1> ~ <mask_32>的对应大模型的隐藏层特征用于mask训练, 使用一个mask_head将其映射成图像大小的预测mask
    - 提取<cls>的对应大模型的隐藏层特征用于分类任务, 使用一个cls_head将其映射成一个0~1的logits值
    - 分析大模型的输出，计算文本输出损失、mask输出损失、cls输出损失
  - prompt:
    - User: 请分析下面这张图片的真伪<image>, 用json格式输出
    - Assistant: "{"result": real<cls>, "mask": [[x1, x2, x3...], [y1, y2, y3], ...]<mask_1><mask_2>...<mask32>}"
  - 微调部分: 
    - mask_head, cls_head, embedding, lm_head

# 实验过程的问题
1. 如果训练太多伦次, 输出结果的语义信息就会丢失

3b模型
2025-04-10 03:54:59 - INFO - last_model_checkpoint: /pubdata/yuyangxin/swift-demo/output/Qwen2.5-VL-3B-Instruct_2025-04-09_23-08/v0-20250409-230822/checkpoint-4334
2025-04-10 03:54:59 - INFO - best_model_checkpoint: /pubdata/yuyangxin/swift-demo/output/Qwen2.5-VL-3B-Instruct_2025-04-09_23-08/v0-20250409-230822/checkpoint-2364

7b模型
2025-04-09 22:31:55 - INFO - last_model_checkpoint: /data0/yuyangxin/finetune-qwen/output/Qwen2.5-VL-7B-Instruct_2025-04-09_12-31/v0-20250409-123145/checkpoint-4785
2025-04-09 22:31:55 - INFO - best_model_checkpoint: /data0/yuyangxin/finetune-qwen/output/Qwen2.5-VL-7B-Instruct_2025-04-09_12-31/v0-20250409-123145/checkpoint-3828

# 总结的问题
1. special token 如果不在指定位置插入, 则模型无法捕获有效的特征
2. 不pad的性能更好

# TODO:
0. 测试sigmoid开启和关闭对性能的影响; 结论: 打开相对好
1. 指导文本的内容出现较为明显的问题, 如果使用隐藏层最后一层的输出则无法完全捕获特征, 是否使用cls分类头进行预测
2. 测试使用隐藏层的使用内容
3. 测试lora_alpha为32和为8对性能的影响
4. 分步骤进行训练, 模型具备基本取证能力 + SAM模型具备取证能力 + 指令监督微调

5. 分步骤进行训练, 模型具备基本取证能力 + SAM模型具备取证能力 + 指令监督微调

6. 添加一些纯结论的数据集进行train试一试是否可行

# 目前发现的问题
1. R+F -> 导致模型输出全是Real, 而没有伪造识别的功能?

# 实现相应的内容
1. hidden_state的最后一层还是取中间层? 或者有没有什么其他方式可以辅助生成? --> 中间层和隐藏层的消融分析
2. 专家模型特征的加入与性能表现
3. 多阶段训练性能分析: 第一阶段 +  第二阶段 + 第三阶段
4. 
5. |<cls>|token的内容是否是真的有效? 结构如何设置? 


# 实现Prompt的相关内容
2. 偷懒 + 加一些上下文的提问和关注
3
