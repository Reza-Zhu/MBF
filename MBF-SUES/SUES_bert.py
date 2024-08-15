import os
import torch
from utils import create_dir, get_yaml_value
from pytorch_pretrained_bert import BertTokenizer, BertModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class Word_Embeding:
    def __init__(self):
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased').to(device)
        self.model = BertModel.from_pretrained('bert-base-uncased')
        self.model.eval()

    def word_embedding(self, text):

        marked_text = text
        tokenized_text = self.tokenizer.tokenize(marked_text).to(device)
        indexed_tokens = self.tokenizer.convert_tokens_to_ids(tokenized_text)
        tokens_tensor = torch.tensor([indexed_tokens]).to(device)
        segments_ids = [1] * len(tokenized_text)
        segments_tensors = torch.tensor([segments_ids]).to(device)

        with torch.no_grad():
            encoded_layers, _ = self.model(tokens_tensor, segments_tensors)

        sentence_embedding = torch.mean(encoded_layers[11], 1)

        return sentence_embedding



wd = Word_Embeding()

heights = [150, 200, 250, 300]
angles = [45, 50, 60, 70]



param = get_yaml_value("settings.yaml")
train_path = os.path.join(param["dataset_path"], "Training")
test_path = os.path.join(param["dataset_path"], "Testing")

for i in range(len(heights)):

    train_height_path = os.path.join(train_path, str(heights[i]))
    test_height_path = os.path.join(test_path, str(heights[i]))

    train_drone_text_path = os.path.join(train_height_path, "text_drone")
    train_satellite_text_path = os.path.join(test_height_path, "text_satellite")

    if not os.path.exists(train_drone_text_path):
        os.mkdir(train_drone_text_path)
    if not os.path.exists(train_satellite_text_path):
        os.mkdir(train_satellite_text_path)

    test_drone_text_path = os.path.join(train_height_path, "text_drone")
    test_satellite_text_path = os.path.join(test_height_path, "text_satellite")

    if not os.path.exists(test_drone_text_path):
        os.mkdir(test_drone_text_path)
    if not os.path.exists(test_satellite_text_path):
        os.mkdir(test_satellite_text_path)

    drone = "The altitude of the drone is %d meters, the angle of camera is %d degree" % (heights[i], angles[i])
    drone_tensor = wd.word_embedding(drone)
    torch.save(drone_tensor, os.path.join(train_drone_text_path, "drone.pth"))
    torch.save(drone_tensor, os.path.join(test_drone_text_path, "drone.pth"))
    # print(os.path.join(train_path, "text_drone", "image-%02d.pth" % (i + 1)))

    # satellite
    satellite = "The altitude of the satellite is 1000 kilometers"
    satellite_tensor = wd.word_embedding(satellite)
    torch.save(satellite_tensor, os.path.join(train_satellite_text_path,  "satellite.pth"))
    torch.save(satellite_tensor, os.path.join(test_satellite_text_path,  "satellite.pth"))


print("Done")

