from transformers.models.bert import BertPreTrainedModel, BertModel
from torch.nn import CrossEntropyLoss, MSELoss
import torch.nn as nn
import torch

class BertForSequenceClassification(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.multi_dropout = nn.ModuleList(
            [nn.Dropout(config.hidden_dropout_prob) for _ in range(config.multi_dropout)])
        self.classifier = nn.Linear(config.hidden_size * 3, config.num_labels)
        self.loss_fct = CrossEntropyLoss()
        self.init_weights()

    # @add_start_docstrings_to_model_forward(BERT_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    # @add_code_sample_docstrings(
    #     tokenizer_class=_TOKENIZER_FOR_DOC,
    #     checkpoint=_CHECKPOINT_FOR_DOC,
    #     output_type=SequenceClassifierOutput,
    #     config_class=_CONFIG_FOR_DOC,
    # )
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for computing the sequence classification/regression loss. Indices should be in :obj:`[0, ...,
            config.num_labels - 1]`. If :obj:`config.num_labels == 1` a regression loss is computed (Mean-Square loss),
            If :obj:`config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        pooled_output = outputs[1]
        hidden_mean_pool = torch.mean(outputs[2][-1], dim=1)
        hidden_max_pool, _ = torch.max(outputs[2][-1], dim=1)

        # pooled_output = self.dropout(pooled_output)
        # logits = self.classifier(pooled_output)

        if labels is not None :
            for i, dropout in enumerate(self.multi_dropout) :
                logits = self.classifier(
                    dropout(
                        torch.cat([pooled_output, hidden_mean_pool, hidden_max_pool], dim=-1)))
                if i == 0 :
                    # outputs = (logits,) + outputs[2 :]  # add hidden states and attention if they are here
                    loss = self.loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
                else :
                    loss = loss + self.loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            output = (loss,) + (logits,)
        else :
            logits = self.classifier(
                torch.cat([pooled_output, hidden_mean_pool, hidden_max_pool], dim=-1))
            output = (logits,)

        return output