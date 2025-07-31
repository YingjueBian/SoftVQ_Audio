from datasets import load_from_disk

try:
    # 加载数据集
    dataset = load_from_disk("/zoomai/colddata/asr/dean/sound_dataset_colddata/laion-audio-300m-arrow_test")
    
    # 检查是否存在train split
    if "train" in dataset:
        train_dataset = dataset["train"]
        # 获取第一条数据
        first_example = train_dataset[0]
        print("数据集中train split的第一条内容:")
        for key, value in first_example.items():
            print(f"{key}: {value}")
    else:
        print("数据集中不存在名为'train'的split.")

except Exception as e:
    print(f"加载数据集时出错: {e}")    