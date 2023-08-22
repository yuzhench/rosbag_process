from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import AutoConfig, AutoModel
from transformers import SegformerFeatureExtractor, SegformerForSemanticSegmentation, TrainingArguments, Trainer
import torch


def main():
    model = SegformerForSemanticSegmentation.from_pretrained("segformer-b1-segments-outputs/checkpoint-160000")
    dummy_input = torch.randn(1, 3, 512, 512)
    torch.onnx.export(model, dummy_input, "model.onnx")


if __name__ == '__main__':
    main()
