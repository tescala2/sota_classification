from models.resnet import vanilla, cbam, sasa, saaa, aacn, botnet, bat
from models.transformers import vit, maxvit, nextvit, swint, deit, deit_ls, cait
from models import convnext, patchconv

models = {
    "resnet18_vanilla": vanilla.resnet18,
    "resnet34_vanilla": vanilla.resnet34,
    "resnet50_vanilla": vanilla.resnet50,
    "resnet101_vanilla": vanilla.resnet101,
    "resnet152_vanilla": vanilla.resnet152,
    "resnet18_cbam": cbam.resnet18_cbam,
    "resnet34_cbam": cbam.resnet34_cbam,
    "resnet50_cbam": cbam.resnet50_cbam,
    "resnet101_cbam": cbam.resnet101_cbam,
    "resnet152_cbam": cbam.resnet152_cbam,
    "resnet18_sasa": sasa.resnet18_sasa,
    "resnet34_sasa": sasa.resnet34_sasa,
    "resnet50_sasa": sasa.resnet50_sasa,
    "resnet101_sasa": sasa.resnet101_sasa,
    "resnet152_sasa": sasa.resnet152_sasa,
    "resnet18_saaa": saaa.resnet18_saaa,
    "resnet34_saaa": saaa.resnet34_saaa,
    "resnet50_saaa": saaa.resnet50_saaa,
    "resnet101_saaa": saaa.resnet101_saaa,
    "resnet152_saaa": saaa.resnet152_saaa,
    "resnet18_aacn": aacn.resnet18_aacn,
    "resnet34_aacn": aacn.resnet34_aacn,
    "resnet50_aacn": aacn.resnet50_aacn,
    "resnet101_aacn": aacn.resnet101_aacn,
    "resnet152_aacn": aacn.resnet152_aacn,
    "resnet18_bot": botnet.resnet18_bot,
    "resnet34_bot": botnet.resnet34_bot,
    "resnet50_bot": botnet.resnet50_bot,
    "resnet101_bot": botnet.resnet101_bot,
    "resnet152_bot": botnet.resnet101_bot,
    "resnet18_bat": bat.resnet18_bat,
    "resnet34_bat": bat.resnet34_bat,
    "resnet50_bat": bat.resnet50_bat,
    "resnet101_bat": bat.resnet101_bat,
    "resnet152_bat": bat.resnet152_bat,
    "vit_b_16": vit.vit_b_16,
    "vit_b_32": vit.vit_b_32,
    "vit_l_16": vit.vit_l_16,
    "vit_l_32": vit.vit_l_32,
    "maxvit_t": maxvit.maxvit_t,
    "nextvit_s": nextvit.nextvit_small,
    "nextvit_b": nextvit.nextvit_base,
    "nextvit_l": nextvit.nextvit_large,
    "swin_t": swint.swin_t,
    "swin_s": swint.swin_s,
    "swin_b": swint.swin_b,
    "swin_v2_t": swint.swin_v2_t,
    "swin_v2_s": swint.swin_v2_s,
    "swin_v2_b": swint.swin_v2_b,
    "convnext_t": convnext.convnext_tiny,
    "convnext_s": convnext.convnext_small,
    "convnext_b": convnext.convnext_base,
    "convnext_l": convnext.convnext_large,
    "S60": patchconv.S60,
    "S120": patchconv.S120,
    "B60": patchconv.B60,
    "B120": patchconv.B120,
    "L60": patchconv.L60,
    "L120": patchconv.L120,
    "deit_tiny_patch16_224": deit.deit_tiny_patch16_224,
    "deit_small_patch16_224": deit.deit_small_patch16_224,
    "deit_base_patch16_224": deit.deit_base_patch16_224,
    "deit_tiny_distilled_patch16_224": deit.deit_tiny_distilled_patch16_224,
    "deit_small_distilled_patch16_224": deit.deit_small_distilled_patch16_224,
    "deit_base_distilled_patch16_224": deit.deit_base_distilled_patch16_224,
    "deit_tiny_patch16_LS": deit_ls.deit_tiny_patch16_LS,
    "deit_small_patch16_LS": deit_ls.deit_small_patch16_LS,
    "deit_medium_patch16_LS": deit_ls.deit_medium_patch16_LS,
    "deit_base_patch16_LS": deit_ls.deit_base_patch16_LS,
    "deit_large_patch16_LS": deit_ls.deit_large_patch16_LS,
    "deit_huge_patch14_LS": deit_ls.deit_huge_patch14_LS,
    "cait_S24_224": cait.cait_S24_224,
    "cait_XXS24_224": cait.cait_XXS24_224,
    "cait_XXS36_224": cait.cait_XXS36_224,
}


def get_model(num_classes, model, pretrained=False):
    return models[model](num_classes=num_classes, pretrained=pretrained)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == '__main__':
    model = get_model(num_classes=50, model='resnet50_vanilla', pretrained=False)
    print(model)
