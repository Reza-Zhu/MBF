import os
import torch
from utils import create_dir, get_yaml_value
from pytorch_pretrained_bert import BertTokenizer, BertModel

class Word_Embeding:
    def __init__(self):
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.model = BertModel.from_pretrained('bert-base-uncased').cuda()
        self.model.eval()

    def word_embedding(self, text):
        # tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        # text = "After stealing money from the bank vault, the bank robber was seen fishing on the Mississippi river bank."
        marked_text = text
        tokenized_text = self.tokenizer.tokenize(marked_text)
        print(tokenized_text)
        indexed_tokens = self.tokenizer.convert_tokens_to_ids(tokenized_text)
        tokens_tensor = torch.tensor([indexed_tokens]).cuda()
        segments_ids = [1] * len(tokenized_text)
        print(segments_ids)
        segments_tensors = torch.tensor([segments_ids]).cuda()
        # Load pre-trained model (weights)
        # model = BertModel.from_pretrained('bert-base-uncased').cuda()
        # Put the model in "evaluation" mode, meaning feed-forward operation.
        with torch.no_grad():
            encoded_layers, _ = self.model(tokens_tensor, segments_tensors)

        # print("Number of layers:", len(encoded_layers))
        # layer_i = 0
        # print("Number of batches:", len(encoded_layers[layer_i]))
        # batch_i = 0
        # print("Number of tokens:", len(encoded_layers[layer_i][batch_i]))
        # token_i = 0
        # print("Number of hidden units:", len(encoded_layers[layer_i][batch_i][token_i]))

        sentence_embedding = torch.mean(encoded_layers[11], 1)
        # print(encoded_layers[11].shape)
        # print(sentence_embedding.shape)
        # print("Our final sentence embedding vector of shape:", sentence_embedding[0].shape[0])
        return sentence_embedding


param = get_yaml_value("settings.yaml")
train_path = os.path.join(param["dataset_path"], "train")
test_path = os.path.join(param["dataset_path"], "test")

wd = Word_Embeding()

# calculate image height from 256m - 121.5m
coff = (256 - 121.5)/53
heights = [256 - coff*i for i in range(1, 54)]
heights.insert(0, 256)
print("image-%02d" % 1)


create_dir(os.path.join(train_path, "text_drone"))
create_dir(os.path.join(test_path, "text_drone"))
create_dir(os.path.join(train_path, "text_satellite"))
create_dir(os.path.join(test_path, "text_satellite"))


# drone
for i in range(54):
    drone = "The altitude of the drone is %d meters" % heights[i]
    drone_tensor = wd.word_embedding(drone)
    torch.save(drone_tensor, os.path.join(train_path, "text_drone", "image-%02d.pth" % (i + 1)))
    torch.save(drone_tensor, os.path.join(test_path, "text_drone", "image-%02d.pth" % (i + 1)))
    print(os.path.join(train_path, "text_drone", "image-%02d.pth" % (i + 1)))


# satellite
satellite = "The altitude of the satellite is 1000 kilometers"
satellite_tensor = wd.word_embedding(satellite)
torch.save(satellite_tensor, os.path.join(train_path, "text_satellite", "satellite.pth"))
torch.save(satellite_tensor, os.path.join(test_path, "text_satellite", "satellite.pth"))




