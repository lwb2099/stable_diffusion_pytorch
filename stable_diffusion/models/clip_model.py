

from transformers import CLIPTextModel, CLIPTokenizer



class CLIPModel:
    def __init__(self, 
                 text_encoder: CLIPTextModel,
                 tokenizer: CLIPTokenizer,
                 max_seq_len: int=77,
                 context_dim: int=768,
                 ):
        """main class"""
        super().__init__()
        self.max_seq_len = max_seq_len
        self.context_dim = context_dim
        self.text_encoder = text_encoder
        self.tokenizer = tokenizer
    
    def tokenize(self, prompt: str="", max_length: int=None, padding: str="max_length", truncation: bool=True):
        return self.tokenizer(prompt, 
                              return_tensors="pt", 
                              padding=padding,
                              truncation=truncation, 
                              max_length=(self.max_seq_len if max_length is None else max_length),)

    def encode_text(self, text):
        if isinstance(text, str):
            text = self.tokenize(text)
        return self.text_encoder(text.input_ids)[0]
    