from models.resnet import vanilla, cbam, sasa, saaa, aacn
from models.transformers import vit, maxvit, nextvit, swint

models = {
    'resnet18_vanilla': vanilla.resnet18,
    'resnet34_vanilla': vanilla.resnet34,
    'resnet50_vanilla': vanilla.resnet50,
    'resnet101_vanilla': vanilla.resnet101,
    'resnet152_vanilla': vanilla.resnet152,
    'resnet18_cbam': cbam.resnet18_cbam,
    'resnet34_cbam': cbam.resnet34_cbam,
    'resnet50_cbam': cbam.resnet50_cbam,
    'resnet101_cbam': cbam.resnet101_cbam,
    'resnet152_cbam': cbam.resnet152_cbam,
    'resnet18_sasa': sasa.resnet18_sasa,
    'resnet34_sasa': sasa.resnet34_sasa,
    'resnet50_sasa': sasa.resnet50_sasa,
    'resnet101_sasa': sasa.resnet101_sasa,
    'resnet152_sasa': sasa.resnet152_sasa,
    'resnet18_saaa': saaa.resnet18_saaa,
    'resnet34_saaa': saaa.resnet34_saaa,
    'resnet50_saaa': saaa.resnet50_saaa,
    'resnet101_saaa': saaa.resnet101_saaa,
    'resnet152_saaa': saaa.resnet152_saaa,
    'resnet18_aacn': aacn.resnet18_aacn,
    'resnet34_aacn': aacn.resnet34_aacn,
    'resnet50_aacn': aacn.resnet50_aacn,
    'resnet101_aacn': aacn.resnet101_aacn,
    'resnet152_aacn': aacn.resnet152_aacn,
    'vit_b_16': vit.vit_b_16,
    'vit_b_32': vit.vit_b_32,
    'vit_l_16': vit.vit_l_16,
    'vit_l_32': vit.vit_l_32,
    'maxvit_t': maxvit.maxvit_t,
    'nextvit_small': nextvit.nextvit_small,
    'nextvit_base': nextvit.nextvit_base,
    'nextvit_large': nextvit.nextvit_large,
    "swin_t": swint.swin_t,
    "swin_s": swint.swin_s,
    "swin_b": swint.swin_b,
    "swin_v2_t": swint.swin_v2_t,
    "swin_v2_s": swint.swin_v2_s,
    "swin_v2_b": swint.swin_v2_b
}


def get_model(num_classes, configs=None):
    model = models[configs['model']](num_classes=num_classes)
    return model


if __name__ == '__main__':
    configs = {
        'model': 'resnet18_vanilla'
    }

    model = get_model(50, configs=configs)
    print(model)
