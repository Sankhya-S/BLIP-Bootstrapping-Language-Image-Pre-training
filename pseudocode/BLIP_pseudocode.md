# BLIP Architecture Pseudocode
```bibtex
class BLIP_MED:  # Multimodal mixture of Encoder-Decoder
    def __init__(self):
        # Step 1: Initialize Vision Encoder
        # Uses ViT (Vision Transformer)
        self.image_encoder = VisionTransformer(
            patch_size=16,  # Divides image into 16x16 patches
            embed_dim=768,  # ViT-Base configuration
            add_cls_token=True  # [CLS] token for global image features
        )
        
        # Step 2: Initialize Text Components
        # Shared components between encoder and decoder 
        self.text_embeddings = TextEmbeddings()  # BERT-style text embeddings
        self.cross_attention = CrossAttentionLayer()  # For image-text interaction
        self.feed_forward = FeedForwardNetwork()  # Standard transformer FFN
        
        # Step 3: Initialize Different Self-Attention Types
        # Key architectural difference for encoding vs decoding 
        self.bidirectional_self_attention = BidirectionalSelfAttention()  # BERT-style
        self.causal_self_attention = CausalSelfAttention()  # GPT-style

    # Step 4: Implement Three Core Functionalities 
    def unimodal_encoder(self, image, text):
        """Mode 1: Separate encoding for Image-Text Contrastive (ITC) learning
           Aligns vision and language representations in shared space"""
        img_feat = self.image_encoder(image)  # Get [CLS] token for image
        txt_feat = self.text_encoder(text)    # Get [CLS] token for text
        return img_feat, txt_feat  # Used for contrastive learning

    def image_grounded_text_encoder(self, image, text):
        """Mode 2: Cross-modal encoding for Image-Text Matching (ITM)
           Learns fine-grained alignment between image and text"""
        # Get image features
        img_feat = self.image_encoder(image)
        # Prepare text with special token
        text = "[Encode] + " + text  
        
        # Process through transformer layers
        for layer in range(num_layers):
            # Step 1: Text self-attention (bidirectional for understanding)
            text_features = self.bidirectional_self_attention(text)
            # Step 2: Cross-attention with image features
            text_features = self.cross_attention(text_features, img_feat)
            # Step 3: Feed-forward processing
            text_features = self.feed_forward(text_features)
            
        return text_features  # Returns [Encode] token output for ITM

    def image_grounded_text_decoder(self, image, text=None):
        """Mode 3: Generative decoding for Language Modeling (LM)
           Enables text generation conditioned on images"""
        # Get image features
        img_feat = self.image_encoder(image)
        # Prepare text with special token for generation
        text = "[Decode] + " + (text if text else "")
        
        # Process through transformer layers
        for layer in range(num_layers):
            # Step 1: Text self-attention (causal for generation)
            text_features = self.causal_self_attention(text)
            # Step 2: Cross-attention with image features
            text_features = self.cross_attention(text_features, img_feat)
            # Step 3: Feed-forward processing
            text_features = self.feed_forward(text_features)
            
        return text_features  # For next token prediction

# Step 5: CapFilt Implementation 
class CapFilt:
    """Implements the Captioning and Filtering bootstrapping method"""
    def __init__(self):
        self.base_model = BLIP_MED()
        self.captioner = None  # Will generate synthetic captions
        self.filter = None     # Will filter noisy captions

    def initialize_modules(self):
        """Initialize and finetune captioner and filter"""
        # Create two copies of base model
        self.captioner = copy.deepcopy(self.base_model)
        self.filter = copy.deepcopy(self.base_model)
        
        # Finetune each for specific tasks
        self.captioner.finetune(coco_dataset, objective='LM')  # For generation
        self.filter.finetune(coco_dataset, objective=['ITC', 'ITM'])  # For matching

    def bootstrap_dataset(self, web_images, web_texts):
        """Main bootstrapping process"""
        # Step 1: Generate synthetic captions
        synthetic_captions = []
        for image in web_images:
            # Use nucleus sampling for diverse captions 
            caption = self.captioner.generate(image, sampling='nucleus', p=0.9)
            synthetic_captions.append(caption)

        # Step 2: Filter captions
        filtered_data = []
        for image, web_text, syn_caption in zip(web_images, web_texts, synthetic_captions):
            # Filter both original and synthetic captions
            if self.filter.matches(image, web_text):
                filtered_data.append((image, web_text))
            if self.filter.matches(image, syn_caption):
                filtered_data.append((image, syn_caption))

        return filtered_data

# Step 6: Training Process
def train_BLIP():
    """Main training loop with three objectives"""
    # Define training objectives
    def compute_itc_loss(img_feat, txt_feat):
        """Contrastive loss with momentum distillation"""
        # Implements contrastive learning 
        pass

    def compute_itm_loss(img_txt_feat):
        """Binary classification with hard negative mining"""
        # Implements matching 
        pass

    def compute_lm_loss(generated_text, target_text):
        """Language modeling with label smoothing"""
        # Implements generation 
        pass

    # Training loop combining all objectives
    for epoch in range(num_epochs):
        for batch in dataloader:
            # Forward pass using all three functionalities
            img_feat, txt_feat = model.unimodal_encoder(batch)
            img_txt_feat = model.image_grounded_text_encoder(batch)
            generated = model.image_grounded_text_decoder(batch)

            # Compute combined loss
            loss = (compute_itc_loss(img_feat, txt_feat) + 
                   compute_itm_loss(img_txt_feat) + 
                   compute_lm_loss(generated, batch.target))
            
            # Update model
            loss.backward()
            optimizer.step()

```