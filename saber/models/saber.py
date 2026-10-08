import torch.nn as nn
import torch.nn.functional as F


class SABER(nn.Module):
    def __init__(self, config):
        super(SABER, self).__init__()
        self.config = config
        self.num_cell_types = config["num_cell_types"]
        self.gene_dim = config["gene_dim"]

        self.prop_dim = config.get("prop_dim", 64)
        self.encoder_hidden_dim = config.get("encoder_hidden_dim", 128)

        dropout_rate = config.get("dropout_rate", 0.1)
        self.encoder = ProportionEncoder(
            input_dim=self.gene_dim,
            prop_dim=self.prop_dim,
            hidden_dim=self.encoder_hidden_dim,
            dropout_rate=dropout_rate,
        )

        self.proportion_head = ProportionHead(
            prop_dim=self.prop_dim,
            num_cell_types=self.num_cell_types,
            hidden_dim=config.get("proportion_head_hidden_dim", 256),
            dropout_rate=dropout_rate,
        )

    def forward(self, x):
        """
        Args:
            x: analysis-scale bulk expression [batch_size, gene_dim]

        Returns:
            proportions: predicted cell proportions [batch_size, num_cell_types]
            scores: proportion logits [batch_size, num_cell_types]
            z_prop: sample-level proportion latent [batch_size, prop_dim]
        """
        z_prop = self.encoder(x)
        proportions, scores = self.proportion_head(z_prop)
        return proportions, scores, z_prop

    def unfreeze_encoder(self):
        for param in self.encoder.parameters():
            param.requires_grad = True

    def unfreeze_proportion_head(self):
        for param in self.proportion_head.parameters():
            param.requires_grad = True

    def configure_encoder_last_linear_finetune(self):
        for param in self.parameters():
            param.requires_grad = False
        last_layer = self.encoder.network[-1]
        if not isinstance(last_layer, nn.Linear):
            raise ValueError(
                "encoder_last_linear requires ProportionEncoder.network[-1] "
                "to be an nn.Linear layer."
            )
        adapt_params = list(last_layer.parameters())
        for param in adapt_params:
            param.requires_grad = True
        return adapt_params

    def get_main_parameters(self):
        return list(self.encoder.parameters()) + list(self.proportion_head.parameters())


class ProportionEncoder(nn.Module):
    """
    Lightweight encoder that produces one sample-level proportion latent.
    """

    def __init__(self, input_dim, prop_dim, hidden_dim=128, dropout_rate=0.1):
        super(ProportionEncoder, self).__init__()
        self.prop_dim = prop_dim

        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, prop_dim),
        )

    def forward(self, x):
        return self.network(x)


class ProportionHead(nn.Module):
    def __init__(self, prop_dim, num_cell_types, hidden_dim=256, dropout_rate=0.1):
        super(ProportionHead, self).__init__()
        self.num_cell_types = num_cell_types
        self.prop_dim = prop_dim

        self.backbone = nn.Sequential(
            nn.Linear(prop_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
        )
        self.output_layer = nn.Linear(hidden_dim, num_cell_types)

    def forward(self, z_prop):
        h = self.backbone(z_prop)
        scores = self.output_layer(h)
        proportions = F.softmax(scores, dim=1)
        return proportions, scores
