import math
import torch
from torch import nn


class SkipBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride, add_dropout, eps=1e-5):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, stride, stride, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels, eps=eps)
        self.activation1 = nn.LeakyReLU()

        self.conv2 = nn.Conv2d(out_channels, out_channels, (3, 3), padding=(1, 1), bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels, eps=eps)
        self.activation2 = nn.LeakyReLU()

        self.conv3 = nn.Conv2d(out_channels, out_channels, (3, 3), padding=(1, 1), bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels, eps=eps)
        self.activation3 = nn.LeakyReLU()

        if add_dropout:
            self.dropout = nn.Dropout(0.2)
        else:
            self.dropout = None

    def forward(self, x):
        skip = self.conv1(x)
        skip = self.bn1(skip)
        skip = self.activation1(skip)

        out = self.conv2(skip)
        out = self.bn2(out)
        out = self.activation2(out)

        if self.dropout is not None:
            out = self.dropout(out)

        out = self.conv3(out)
        out = self.bn3(out)
        out = self.activation3(out)

        return skip + out


class TransformerEncoderTable(nn.Module):

    def __init__(self,
                 num_heads=4,
                 dff=1024,
                 num_layers=4,
                 d_model=256,
                 num_clustering_heads=1,
                 is_use_image_patches=False,
                 is_use_4_points=False,
                 max_doc_size=1024,
                 image_patches_channels=1,
                 is_sum_embeddings=True,
                 use_content_emb=True,
                 out_dim=300):
        super().__init__()
        print('d_model', d_model)
        print('is_sum_embeddings', is_sum_embeddings)
        print('use_content_emb', use_content_emb)
        self.num_clustering_heads = num_clustering_heads
        self.is_use_image_patches = is_use_image_patches
        self.is_use_4_points = is_use_4_points
        self.is_sum_embeddings = is_sum_embeddings
        self.image_patches_channels = image_patches_channels
        self.use_content_emb = use_content_emb

        self.OUT_DIM = out_dim
        num_coords = 8 if self.is_use_4_points else 4
        print('num_coords', num_coords)

        if self.is_sum_embeddings:
            self.PATCH_EMBEDDING_SIZE = d_model
        else:
            COORD_EMBEDDING_SIZE = 64
            WORD_CONTENT_EMBEDDING_SIZE = 128
            self.PATCH_EMBEDDING_SIZE = 128
            if self.is_use_4_points:
                COORD_EMBEDDING_SIZE = 32
            d_model = COORD_EMBEDDING_SIZE * num_coords + WORD_CONTENT_EMBEDDING_SIZE
            if self.is_use_image_patches:
                d_model += self.PATCH_EMBEDDING_SIZE

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=dff,
            dropout=0.1,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers)

        self.activation_clustering = nn.GELU()
        self.dropout = nn.Dropout(0.1)
        self.layer_norm = nn.LayerNorm(self.OUT_DIM)

        self.dense_layer_clustering = []
        self.out_layer_clustering = []
        self.layer_norms = []
        for _ in range(self.num_clustering_heads):
            self.dense_layer_clustering.append(nn.Linear(
                d_model, self.OUT_DIM))
            self.out_layer_clustering.append(
                nn.Linear(self.OUT_DIM, self.OUT_DIM))
            self.layer_norms.append(nn.LayerNorm(self.OUT_DIM))

        self.dense_layer_clustering = nn.ModuleList(
            self.dense_layer_clustering)
        self.out_layer_clustering = nn.ModuleList(self.out_layer_clustering)
        self.layer_norms = nn.ModuleList(self.layer_norms)

        if self.is_sum_embeddings:
            self.pos_emb = nn.ModuleList([
                nn.Embedding(max_doc_size, d_model) for _ in range(num_coords)
            ])
            if self.use_content_emb:
                self.word_content_emb = nn.Embedding(30016, d_model)
            self.embedding_norm = nn.LayerNorm(d_model)
            self.embedding_dropout = nn.Dropout(0.1)
        else:
            self.pos_emb = nn.Embedding(int(max_doc_size),
                                        COORD_EMBEDDING_SIZE)
            self.word_content_emb = nn.Embedding(30016,
                                                 WORD_CONTENT_EMBEDDING_SIZE)

        if self.is_use_image_patches:
            self.block1 = SkipBlock(self.image_patches_channels, 16, (1, 1),
                                    False)
            self.block2 = SkipBlock(16, 32, (2, 2), False)
            self.block3 = SkipBlock(32, 48, (2, 2), True)
            self.block4 = SkipBlock(48, 80, (1, 1), True)
            self.patch_linear_emb = nn.Linear(80 * 8 * 8,
                                              self.PATCH_EMBEDDING_SIZE)
            if not self.is_sum_embeddings:
                self.norm = nn.LayerNorm(self.PATCH_EMBEDDING_SIZE)

        # Xavier uniform initialization:
        self._reset_parameters()

    def _reset_parameters(self):
        r"""Initiate parameters in the transformer model."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self,
                word_boxes,
                contents_idx,
                padding_mask,
                img_patches=None):
        if padding_mask is not None:
            # (batch, seq_len)
            padding_mask = (padding_mask == 0)

        if self.is_use_image_patches:
            # <batch_size, number_of_words * image_patches_channels, 32, 32>
            dim1, number_of_patches, dim3, dim4 = img_patches.shape
            number_of_words = number_of_patches // self.image_patches_channels
            # <batch_size, number_of_words, image_patches_channels, 32, 32>
            img_patches = torch.reshape(
                img_patches, (dim1, number_of_words,
                              self.image_patches_channels, dim3, dim4))
            # <batch_size * number_of_words, 3, 32, 32>
            img_patches = torch.reshape(
                img_patches, (dim1 * number_of_words,
                              self.image_patches_channels, dim3, dim4))

            # (batch_size * number_of_words, 16, 32, 32)
            patch_emb = self.block1(img_patches)
            # (batch_size * number_of_words, 32, 16, 16)
            patch_emb = self.block2(patch_emb)
            # (batch_size * number_of_words, 48, 8, 8)
            patch_emb = self.block3(patch_emb)
            # (batch_size * number_of_words, 80, 8, 8)
            patch_emb = self.block4(patch_emb)

            dim1, dim2, dim3, dim4 = patch_emb.shape
            # (batch_size * number_of_words, 5120)
            patch_emb = torch.reshape(patch_emb, (dim1, dim2 * dim3 * dim4))
            # (batch_size * number_of_words, 128)
            patch_emb = self.patch_linear_emb(patch_emb)
            # (batch_size, number_of_words, 128)
            patch_emb = torch.reshape(
                patch_emb, (-1, number_of_words, self.PATCH_EMBEDDING_SIZE))
            if not self.is_sum_embeddings:
                patch_emb = self.norm(patch_emb)

        if self.is_use_4_points:
            if self.is_sum_embeddings:
                x = 0.0
                size = word_boxes.size()

                if word_boxes.dim() == 4:
                    word_boxes = word_boxes.view(size[0], size[1],
                                                 size[2] * size[3])
                for i, layer in enumerate(self.pos_emb):
                    x += layer(word_boxes[:, :, i])

                if self.use_content_emb:
                    x += self.word_content_emb(contents_idx)

                if self.is_use_image_patches:
                    x += patch_emb
                x = self.embedding_norm(x)
                x = self.embedding_dropout(x)
            else:
                x1 = self.pos_emb(word_boxes[:, :, 0, 0])
                y1 = self.pos_emb(word_boxes[:, :, 0, 1])
                x2 = self.pos_emb(word_boxes[:, :, 1, 0])
                y2 = self.pos_emb(word_boxes[:, :, 1, 1])
                x3 = self.pos_emb(word_boxes[:, :, 2, 0])
                y3 = self.pos_emb(word_boxes[:, :, 2, 1])
                x4 = self.pos_emb(word_boxes[:, :, 3, 0])
                y4 = self.pos_emb(word_boxes[:, :, 3, 1])
                contents_idx_emb = self.word_content_emb(contents_idx)
                x = torch.cat(
                    [x1, y1, x2, y2, x3, y3, x4, y4, contents_idx_emb], dim=-1)
        else:
            if self.is_sum_embeddings:
                x = 0.0
                for i, layer in enumerate(self.pos_emb):
                    x += layer(word_boxes[:, :, i])

                if self.use_content_emb:
                    x += self.word_content_emb(contents_idx)

                if self.is_use_image_patches:
                    x += patch_emb
                x = self.embedding_norm(x)
                x = self.embedding_dropout(x)
            else:
                x1 = self.pos_emb(word_boxes[:, :, 0])
                y1 = self.pos_emb(word_boxes[:, :, 1])
                x2 = self.pos_emb(word_boxes[:, :, 2])
                y2 = self.pos_emb(word_boxes[:, :, 3])

                contents_idx_emb = self.word_content_emb(contents_idx)
                if self.is_use_image_patches:
                    x = torch.cat(
                        [x1, y1, x2, y2, contents_idx_emb, patch_emb], dim=-1)
                else:
                    x = torch.cat([x1, y1, x2, y2, contents_idx_emb], dim=-1)

        x = x.permute(1, 0, 2)
        enc_output = self.encoder(x, src_key_padding_mask=padding_mask)
        # (batch_size, inp_seq_len, d_model)
        enc_output = enc_output.permute(1, 0, 2)

        matmul_qk = []
        for i in range(self.num_clustering_heads):
            # [batch_size, MAX_NUM_OF_WORDS, 1024]
            clustering_output = self.activation_clustering(
                self.dense_layer_clustering[i](enc_output))
            # TODO: switch layer_norm or layer_norms depending on the model trained
            # clustering_output = self.layer_norm(clustering_output)
            clustering_output = self.layer_norms[i](clustering_output)
            clustering_output = self.dropout(clustering_output)
            # [batch_size, MAX_NUM_OF_WORDS, out_dim]
            clustering_output = self.out_layer_clustering[i](clustering_output)

            # [batch_size, MAX_NUM_OF_WORDS, out_dim // 2]
            Q = clustering_output[:, :, :(self.OUT_DIM // 2)]
            # [batch_size, MAX_NUM_OF_WORDS, out_dim // 2]
            K = clustering_output[:, :, (self.OUT_DIM // 2):]
            # [batch_size, MAX_NUM_OF_WORDS, MAX_NUM_OF_WORDS]
            tmp = torch.matmul(Q, K.transpose(-2, -1))
            matmul_qk.append(tmp)

        return matmul_qk
