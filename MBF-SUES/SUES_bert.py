import os
import torch
from pytorch_pretrained_bert import BertTokenizer, BertModel, BertForMaskedLM


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

wd = Word_Embeding()

heights = [150, 200, 250, 300]
angles = [45, 50, 60, 70]
# print("image-%02d" % 1)

train_path = "/home/sues/media/disk2/Datasets/Training"
test_path = "/home/sues/media/disk2/Datasets/Testing"


for i in range(len(heights)):

    train_path = "/home/sues/media/disk2/Datasets/Training"
    test_path = "/home/sues/media/disk2/Datasets/Testing"
    train_path = os.path.join(train_path, str(heights[i]))
    test_path = os.path.join(test_path, str(heights[i]))

    train_drone_text_path = os.path.join(train_path, "text_drone")
    train_satellite_text_path = os.path.join(train_path, "text_satellite")

    if not os.path.exists(train_drone_text_path):
        os.mkdir(train_drone_text_path)
    if not os.path.exists(train_satellite_text_path):
        os.mkdir(train_satellite_text_path)

    test_drone_text_path = os.path.join(test_path, "text_drone")
    test_satellite_text_path = os.path.join(test_path, "text_satellite")

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




