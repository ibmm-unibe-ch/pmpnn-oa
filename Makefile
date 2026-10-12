DATA_URL = https://files.ipd.uw.edu/pub/training_sets/pdb_2021aug02.tar.gz
VANILLA_002_URL = https://github.com/dauparas/ProteinMPNN/raw/refs/heads/main/vanilla_model_weights/v_48_002.pt
VANILLA_010_URL = https://github.com/dauparas/ProteinMPNN/raw/refs/heads/main/vanilla_model_weights/v_48_010.pt
VANILLA_020_URL = https://github.com/dauparas/ProteinMPNN/raw/refs/heads/main/vanilla_model_weights/v_48_020.pt
VANILLA_030_URL = https://github.com/dauparas/ProteinMPNN/raw/refs/heads/main/vanilla_model_weights/v_48_030.pt
DATA_DIR = data
CKPT_DIR = ckpt

.PHONY: env
env:
	uv sync
 
.PHONY: data
data:
	mkdir -p $(DATA_DIR)
	wget -c -P $(DATA_DIR) $(DATA_URL)
	tar -xzf $(DATA_DIR)/pdb_2021aug02.tar.gz -C $(DATA_DIR)

.PHONY: ckpt
ckpt:
	mkdir -p $(CKPT_DIR)
	wget -c -P $(CKPT_DIR) $(VANILLA_002_URL)
	wget -c -P $(CKPT_DIR) $(VANILLA_010_URL)
	wget -c -P $(CKPT_DIR) $(VANILLA_020_URL)
	wget -c -P $(CKPT_DIR) $(VANILLA_030_URL)

.PHONY: clean
clean:
	find . -name "*.pyc" -delete
	find . -name "__pycache__" -delete
	find . -name "*.egg-info" -exec rm -rf {} +