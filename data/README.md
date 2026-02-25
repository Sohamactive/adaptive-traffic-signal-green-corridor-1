# Data Sets

## traffic-vehicles-object-detection
- Sample traffic videos (excluded from git)
- YOLO format annotations
- Data preprocessing scripts
- [Link to Kaggel dataset](https://www.kaggle.com/datasets/saumyapatel/traffic-vehicles-object-detection)
- Download command
```bash
curl -L -o ./data/raw/traffic-vehicles-object-detection.zip https://www.kaggle.com/api/v1/datasets/download/saumyapatel/traffic-vehicles-object-detection\
&&
unzip -o data/raw/traffic-vehicles-object-detection/*.zip -d data/raw/traffic-vehicles-object-detection\
&&
rm data/raw/traffic-vehicles-object-detection/*.zip
```