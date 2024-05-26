import torch
import torch.nn as nn
from transformers.models.gpt2.modeling_gpt2 import GPT2Model,GPT2Config
from transformers import LlamaConfig, LlamaModel, LlamaTokenizer,LlamaForCausalLM,AutoTokenizer,AutoModel,AutoConfig,BertModel,BertConfig
from layers.Embed import DataEmbedding
from transformers import GPT2Tokenizer
from einops import rearrange, repeat
from math import sqrt

class PrefixEncoder(torch.nn.Module):
    r'''
    The torch.nn model to encode the prefix

    Input shape: (batch-size, prefix-length)

    Output shape: (batch-size, prefix-length, 2*layers*hidden)
    '''
    def __init__(self, pre_seq_len,hidden_size,prefix_hidden_size,num_hidden_layers=6,prefix_projection=True):
        super().__init__()
        self.prefix_projection = prefix_projection
        self.pre_seq_len = pre_seq_len
        self.alpha = 0.01
        if self.prefix_projection:
            # Use a two-layer MLP to encode the prefix
            self.embedding = torch.nn.Embedding(pre_seq_len, hidden_size)
            self.trans = torch.nn.Sequential(
                torch.nn.Linear(hidden_size, prefix_hidden_size),
                torch.nn.Tanh(),
                torch.nn.Linear(prefix_hidden_size, num_hidden_layers * 2 * hidden_size)
            )

        else:
            self.embedding = torch.nn.Embedding(pre_seq_len, num_hidden_layers * 2 * hidden_size)
            p_param=0
            for name, param in self.embedding.named_parameters():
                p_param += param.numel()
            print('p param is {}'.format(p_param))
            self.knowledge_trans = torch.nn.Sequential(
                torch.nn.Linear(hidden_size, prefix_hidden_size),
                torch.nn.Tanh(),
                torch.nn.Linear(prefix_hidden_size, num_hidden_layers * 2 * hidden_size))


    def forward(self, prefix, knowledge_embeddings=None):
        if self.prefix_projection:
            prefix_tokens = self.embedding(prefix)
            past_key_values = self.trans(prefix_tokens)
        else:
            past_key_values = self.embedding(prefix)
            if knowledge_embeddings!= None:
                knowledge_embeddings=knowledge_embeddings.repeat(past_key_values.size(0), 1, 1)
                knowledge_past_key_values = self.knowledge_trans(knowledge_embeddings)
                past_key_values = past_key_values + knowledge_past_key_values * self.alpha
        return past_key_values

class Model(nn.Module):

    def __init__(self, configs):
        super(Model, self).__init__()
        self.is_ln = configs.ln
        self.task_name = configs.task_name
        self.pred_len = configs.pred_len
        self.seq_len = configs.seq_len
        self.top_k = configs.top_k
        self.patch_size = configs.patch_size
        self.stride = configs.stride
        self.seq_len = configs.seq_len
        self.d_ff = configs.d_ff
        self.test_mask_rate = configs.test_mask_rate
        self.patch_num = self.seq_len//self.patch_size
        self.device = "cuda:{}".format(configs.gpu)
        self.configs = configs
        self.is_seq_output = self.configs.seq_token > 0


        if not self.configs.use_llama:
            if self.configs.use_bert:
                self.bert_config = BertConfig.from_pretrained('bert')
                self.bert_config.num_hidden_layers = configs.gpt_layers
                self.bert_config.output_attentions = True
                self.bert_config.output_hidden_states = True
                self.gpt2 = BertModel.from_pretrained(
                    'bert',
                    local_files_only=True,
                    config=self.bert_config,
                )
            else:
                # self.gpt2 = GPT2Model.from_pretrained('gpt2/gpt2/', output_attentions=True,
                #                                       output_hidden_states=True, local_files_only=True)
                # self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2/gpt2/', local_files_only=True)
                self.gpt2_config = GPT2Config()
                self.gpt2 = GPT2Model(self.gpt2_config)
                self.gpt2.h = self.gpt2.h[:configs.gpt_layers]
        else:
            self.llama_config = LlamaConfig.from_pretrained(
                'llama-2-7B')
            self.llama_config.output_attentions = True
            self.llama_config.output_hidden_states = True
            self.gpt2 = LlamaModel.from_pretrained(
                "llama-2-7B",
                # 'huggyllama/llama-7b',
                trust_remote_code=True,
                local_files_only=True,
                config=self.llama_config,
                torch_dtype=torch.float32
            )

            self.tokenizer = LlamaTokenizer.from_pretrained(
                "llama-2-7B/tokenizer.model",
                # 'huggyllama/llama-7b',
                trust_remote_code=True,
                local_files_only=True
            )
            self.gpt2.layers = self.gpt2.layers[:configs.gpt_layers]
            configs.d_model = 4096
            self.configs = configs

        self.dropout = nn.Dropout(configs.dropout)
        self.enc_embedding = DataEmbedding(self.patch_size, self.configs.d_model, configs.embed,
                                           configs.freq,
                                           configs.dropout)



        self.device = torch.device(self.device if configs.use_gpu else 'cpu')

        self.contrastive_dim = 128
        self.contrastive_patch_projector = nn.Linear(configs.d_model,self.contrastive_dim)
        self.contrastive_instance_projector = nn.Linear(configs.d_model*self.patch_num,self.contrastive_dim)
        self.instance_W = nn.Parameter(torch.rand(self.contrastive_dim,self.contrastive_dim))
        self.patch_W = nn.Parameter(torch.rand(self.contrastive_dim, self.contrastive_dim))
        self.cross_entropy_loss = nn.CrossEntropyLoss()

        # missing rate token
        if self.configs.cov_prompt:
            self.miss_token = nn.Parameter(torch.rand(1,configs.d_model))
            self.covariable_embedding = nn.Linear(4,configs.d_model)

        if self.configs.output_token:
            self.output_token = nn.Parameter(torch.rand(1,configs.d_model))

        # init prefix tuning
        self.prefix_length = configs.prefix_length

        self.is_prefix_tuning = configs.prefix_tuning
        if self.is_prefix_tuning:
            self.prefix_tokens = nn.Parameter(torch.rand(self.prefix_length,configs.d_model))

        self.is_prefix_tuningv2 = configs.prefix_tuningv2
        if self.is_prefix_tuningv2:
            self.prefix_tokens = torch.arange(self.prefix_length).long()
            self.prefix_encoder = PrefixEncoder(pre_seq_len=self.prefix_length,hidden_size=configs.d_model,prefix_hidden_size=configs.d_model,num_hidden_layers=configs.gpt_layers)


        if not self.configs.frozen_lm:
            if not self.configs.train_all_lm:
                for i, (name, param) in enumerate(self.gpt2.named_parameters()):
                    if 'ln' in name or 'wpe' in name or 'norm' in name or 'LayerNorm' in name or 'position_embeddings' in name:
                        param.requires_grad = True
                    elif ('mlp' in name or 'dense' in name) and configs.mlp == 1:
                        param.requires_grad = True
                    else:
                        param.requires_grad = False
            else:
                for param in self.gpt2.parameters():
                    param.requires_grad = True

        else:
            for param in self.gpt2.parameters():
                param.requires_grad = False

        if configs.use_gpu:
            self.gpt2.to(device=self.device)


        self.ln_proj = nn.LayerNorm(configs.d_model)

        if not self.is_seq_output:
            self.out_layer = nn.Linear(
                configs.d_model*self.patch_num,
                # configs.c_out*self.patch_size,
                self.patch_size*self.patch_num,
                bias=True)
        else:
            self.seq_token = nn.Parameter(torch.rand(self.configs.seq_token,self.configs.d_model))
            self.out_layer = nn.Linear(
                configs.d_model * self.configs.seq_token,
                # configs.c_out*self.patch_size,
                self.patch_size * self.patch_num,
                bias=True)
        if self.configs.alignment:
            self.reprogramming_layer = ReprogrammingLayer(self.configs.d_model, 8, None, self.configs.d_model)
            self.word_embeddings = self.gpt2.get_input_embeddings().weight
            self.vocab_size = self.word_embeddings.shape[0]
            self.num_tokens = 1000
            self.mapping_layer = nn.Linear(self.vocab_size, self.num_tokens)


    def get_prompt(self, batch_size):
        prefix_tokens = self.prefix_tokens.unsqueeze(0).expand(batch_size, -1).to(self.device)
        past_key_values = self.prefix_encoder(prefix_tokens)
        bsz, seqlen, _ = past_key_values.shape
        past_key_values = past_key_values.view(
            bsz,
            seqlen,
            self.configs.gpt_layers * 2,
            12,
            self.configs.d_model // 12
        )
        past_key_values = self.dropout(past_key_values)
        past_key_values = past_key_values.permute([2, 0, 3, 1, 4]).split(2)
        return past_key_values
    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None,skip_output=False):

        dec_out = self.imputation(
            x_enc, x_mark_enc, x_dec, x_mark_dec, mask,skip_output=skip_output)
        return dec_out  # [B, L, D]

    def imputation(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask,skip_output=False):

        if mask is None:
            means = x_enc.mean(1, keepdim=True).detach()
            x_enc = x_enc - means
            stdev = torch.sqrt(
                torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
            x_enc /= stdev
        else:
            # Normalization from Non-stationary Transformer
            means = torch.sum(x_enc, dim=1) / torch.sum(mask == 1, dim=1)
            means = means.unsqueeze(1).detach()
            x_enc = x_enc - means
            x_enc = x_enc.masked_fill(mask == 0, 0)
            stdev = torch.sqrt(torch.sum(x_enc * x_enc, dim=1) /
                               torch.sum(mask == 1, dim=1) + 1e-5)
            stdev = stdev.unsqueeze(1).detach()
            x_enc /= stdev
        B, T, N = x_enc.size()

        if self.configs.cov_prompt:
            x_enc_seg = rearrange(x_enc,'b (patch_num patch_size) n -> (b patch_num) patch_size n',patch_size=self.patch_size)
            mask_seg = rearrange(mask,'b (patch_num patch_size) n -> (b patch_num) patch_size n',patch_size=self.patch_size)
            patch_min_values = torch.min(x_enc_seg, dim=1)[0] # (b patch_num) n
            patch_max_values = torch.max(x_enc_seg, dim=1)[0]
            patch_medians = torch.median(x_enc_seg, dim=1).values
            variable_min_values = torch.min(x_enc, dim=1)[0]
            variable_max_values = torch.max(x_enc, dim=1)[0]
            variable_medians = torch.median(x_enc,dim=1).values

            patch_trends = x_enc_seg.diff(dim=1).sum(dim=1)
            variable_trends = x_enc.diff(dim=1).sum(dim=1)
            patch_missing_rate = 1 - torch.sum(mask_seg,dim=1)/self.patch_size

            patch_feature = torch.cat([patch_min_values.unsqueeze(-1),patch_medians.unsqueeze(-1),patch_max_values.unsqueeze(-1),patch_trends.unsqueeze(-1)],dim=-1)
            variable_feature = torch.cat([variable_min_values.unsqueeze(-1),variable_medians.unsqueeze(-1),variable_max_values.unsqueeze(-1),variable_trends.unsqueeze(-1)],dim=-1)
            patch_feature = rearrange(patch_feature,'(b patch_num) n feature_num -> (b n) patch_num feature_num',patch_num=self.patch_num)
            variable_feature = rearrange(variable_feature,'b n feature_num -> (b n) feature_num').unsqueeze(1)
            overall_feature = torch.cat([variable_feature,patch_feature],dim=1)
            covariable_embedding = self.covariable_embedding(overall_feature)

            missing_embedding = torch.matmul(patch_missing_rate.unsqueeze(-1),self.miss_token)  # (bn)
            missing_embedding = rearrange(missing_embedding,'(b patch_num) n d_model -> (b n) patch_num d_model',patch_num=self.patch_num)


        x_enc = rearrange(x_enc, 'b (patch_num patch_size) c -> (b c) patch_num patch_size', patch_size=self.patch_size)
        patch_num = x_enc.shape[1]


        enc_out = self.enc_embedding(x_enc, None)  # [B,T,C]

        if self.configs.alignment:
            source_embeddings = self.mapping_layer(self.word_embeddings.permute(1, 0)).permute(1, 0)
            enc_out = self.reprogramming_layer(enc_out, source_embeddings, source_embeddings)

        if self.configs.cov_prompt:
            enc_out = enc_out + covariable_embedding[:,1:,:]+ missing_embedding
            enc_out = torch.cat([covariable_embedding[:,:1,:],enc_out],dim=1)

        if self.is_prefix_tuning:
            prefix_tokens = repeat(self.prefix_tokens,'seq_len dmodel -> repeat seq_len dmodel',repeat=enc_out.shape[0])
            gpt2_enc_out = torch.cat([prefix_tokens,enc_out], dim=1)
        else:
            gpt2_enc_out = enc_out


        if self.is_seq_output:
            seq_token = repeat(self.seq_token, 'seq_len dmodel -> repeat seq_len dmodel',
                                   repeat=enc_out.shape[0])
            gpt2_enc_out = torch.cat([gpt2_enc_out,seq_token],dim=1)

        if self.is_prefix_tuningv2:
            past_key_values = self.get_prompt(batch_size=B*N)
            outputs = self.gpt2(inputs_embeds=gpt2_enc_out,past_key_values=past_key_values).last_hidden_state
        else:
            if self.configs.output_token:
                output_token = repeat(self.output_token, 'seq_len dmodel -> repeat seq_len dmodel',
                                       repeat=enc_out.shape[0])
                gpt2_enc_out = torch.cat([gpt2_enc_out, output_token], dim=1)
            outputs = self.gpt2(inputs_embeds=gpt2_enc_out).last_hidden_state

        cls_token = outputs[:, 0, :]

        if not self.is_seq_output:
            outputs = outputs[:,-patch_num:,:]
        else:
            outputs = outputs[:, -self.configs.seq_token:, :]
        outputs = self.ln_proj(outputs)
        if skip_output:
            return outputs

        dec_out = rearrange(outputs,'b patch_num dmodel -> b (patch_num dmodel)')
        dec_out = self.out_layer(dec_out)

        dec_out = rearrange(dec_out, '(b c) seq_len -> b seq_len c', c=N)
        # De-Normalization from Non-stationary Transformer
        dec_out = dec_out * \
                  (stdev[:, 0, :].unsqueeze(1).repeat(
                      1, dec_out.shape[1], 1))
        dec_out = dec_out + \
                  (means[:, 0, :].unsqueeze(1).repeat(
                      1, dec_out.shape[1], 1))
        return dec_out,outputs


class ReprogrammingLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_keys=None, d_llm=None, attention_dropout=0.1):
        super(ReprogrammingLayer, self).__init__()

        d_keys = d_keys or (d_model // n_heads)

        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_llm, d_keys * n_heads)
        self.value_projection = nn.Linear(d_llm, d_keys * n_heads)
        self.out_projection = nn.Linear(d_keys * n_heads, d_llm)
        self.n_heads = n_heads
        self.dropout = nn.Dropout(attention_dropout)

    def forward(self, target_embedding, source_embedding, value_embedding):
        B, L, _ = target_embedding.shape
        S, _ = source_embedding.shape
        H = self.n_heads

        target_embedding = self.query_projection(target_embedding).view(B, L, H, -1)
        source_embedding = self.key_projection(source_embedding).view(S, H, -1)
        value_embedding = self.value_projection(value_embedding).view(S, H, -1)

        out = self.reprogramming(target_embedding, source_embedding, value_embedding)

        out = out.reshape(B, L, -1)

        return self.out_projection(out)

    def reprogramming(self, target_embedding, source_embedding, value_embedding):
        B, L, H, E = target_embedding.shape

        scale = 1. / sqrt(E)

        scores = torch.einsum("blhe,she->bhls", target_embedding, source_embedding)

        A = self.dropout(torch.softmax(scale * scores, dim=-1))
        reprogramming_embedding = torch.einsum("bhls,she->blhe", A, value_embedding)

        return reprogramming_embedding