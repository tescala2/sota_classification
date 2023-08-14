#!/bin/bash

if ! command -v python &> /dev/null
then
  PY_EXE="python3"
else
  PY_EXE="python"
fi

BATCH_SIZE=64

#######################################################################
##  ResNet (Vanilla)  #################################################
#######################################################################

if [[ $1 = "resnet18_vanilla" ]]
then
  $PY_EXE train.py \
-dataset rsid_all_remap_chip \
-name resnet18_vanilla \
-hardware "$2" \
-bs $BATCH_SIZE
elif [[ $1 = "resnet34_vanilla"]]
then
  $PY_EXE train.py \
-dataset mstar \
-name resnet34_vanilla \
-hardware "$2" \
-bs $BATCH_SIZE
elif [[ $1 = "resnet50_vanilla" ]]
then
  $PY_EXE train.py \
-dataset mstar \
-name resnet50_vanilla \
-hardware "$2" \
-bs $BATCH_SIZE
elif [[ $1 = "resnet101_vanilla" ]]
then
  $PY_EXE train.py \
-dataset mstar \
-name resnet101_vanilla \
-hardware "$2" \
-bs $BATCH_SIZE
elif [[ $1 = "resnet152_vanilla" ]]
then
  $PY_EXE train.py \
-dataset mstar \
-name resnet152_vanilla \
-hardware "$2" \
-bs $BATCH_SIZE

#######################################################################
##  ResNet (CBAM)  ####################################################
#######################################################################

elif [[ $1 = "resnet18_cbam" ]]
then
  $PY_EXE train.py \
-dataset mstar \
-name resnet18_cbam \
-hardware "$2" \
-bs $BATCH_SIZE
elif [[ $1 = "resnet34_cbam" ]]
then
  $PY_EXE train.py \
-dataset mstar \
-name resnet34_cbam \
-hardware "$2" \
-bs $BATCH_SIZE
elif [[ $1 = "resnet50_cbam" ]]
then
  $PY_EXE train.py \
-dataset mstar \
-name resnet50_cbam \
-hardware "$2" \
-bs $BATCH_SIZE
elif [[ $1 = "resnet101_cbam" ]]
then
  $PY_EXE train.py \
-dataset mstar \
-name resnet101_cbam \
-hardware "$2" \
-bs $BATCH_SIZE
elif [[ $1 = "resnet152_cbam" ]]
then
  $PY_EXE train.py \
-dataset mstar \
-name resnet152_cbam \
-hardware "$2" \
-bs $BATCH_SIZE

#######################################################################
##  ResNet (SASA)  ####################################################
#######################################################################

elif [[ $1 = "resnet18_sasa" ]]
then
  $PY_EXE train.py \
-dataset mstar \
-name resnet18_sasa \
-hardware "$2" \
-bs $BATCH_SIZE
elif [[ $1 = "resnet34_sasa" ]]
then
  $PY_EXE train.py \
-dataset mstar \
-name resnet34_sasa \
-hardware "$2" \
-bs $BATCH_SIZE
elif [[ $1 = "resnet50_sasa" ]]
then
  $PY_EXE train.py \
-dataset mstar \
-name resnet50_sasa \
-hardware "$2" \
-bs $BATCH_SIZE
elif [[ $1 = "resnet101_sasa" ]]
then
  $PY_EXE train.py \
-dataset mstar \
-name resnet101_sasa \
-hardware "$2" \
-bs $BATCH_SIZE
elif [[ $1 = "resnet152_sasa" ]]
then
  $PY_EXE train.py \
-dataset mstar \
-name resnet152_sasa \
-hardware "$2" \
-bs $BATCH_SIZE

#######################################################################
##  ResNet (SAAA)  ####################################################
#######################################################################

elif [[ $1 = "resnet18_saaa" ]]
then
  $PY_EXE train.py \
-dataset mstar \
-name resnet18_saaa \
-hardware "$2" \
-bs $BATCH_SIZE
elif [[ $1 = "resnet34_saaa" ]]
then
  $PY_EXE train.py \
-dataset mstar \
-name resnet34_saaa \
-hardware "$2" \
-bs $BATCH_SIZE
elif [[ $1 = "resnet50_saaa" ]]
then
  $PY_EXE train.py \
-dataset mstar \
-name resnet50_saaa \
-hardware "$2" \
-bs $BATCH_SIZE
elif [[ $1 = "resnet101_saaa" ]]
then
  $PY_EXE train.py \
-dataset mstar \
-name resnet101_saaa \
-hardware "$2" \
-bs $BATCH_SIZE
elif [[ $1 = "resnet152_saaa" ]]
then
  $PY_EXE train.py \
-dataset mstar \
-name resnet152_saaa \
-hardware "$2" \
-bs $BATCH_SIZE

#######################################################################
##  ResNet (AACN)  ####################################################
#######################################################################

elif [[ $1 = "resnet18_aacn" ]]
then
  $PY_EXE train.py \
-dataset mstar \
-name resnet18_aacn \
-hardware "$2" \
-bs $BATCH_SIZE
elif [[ $1 = "resnet34_aacn" ]]
then
  $PY_EXE train.py \
-dataset mstar \
-name resnet34_aacn \
-hardware "$2" \
-bs $BATCH_SIZE
elif [[ $1 = "resnet50_aacn" ]]
then
  $PY_EXE train.py \
-dataset mstar \
-name resnet50_aacn \
-hardware "$2" \
-bs $BATCH_SIZE
elif [[ $1 = "resnet101_aacn" ]]
then
  $PY_EXE train.py \
-dataset mstar \
-name resnet101_aacn \
-hardware "$2" \
-bs $BATCH_SIZE
elif [[ $1 = "resnet152_aacn" ]]
then
  $PY_EXE train.py \
-dataset mstar \
-name resnet152_aacn \
-hardware "$2" \
-bs $BATCH_SIZE

#######################################################################
##  ResNet (BAT)  #####################################################
#######################################################################

elif [[ $1 = "resnet18_bot" ]]
then
  $PY_EXE train.py \
-dataset mstar \
-name resnet18_bot \
-hardware "$2" \
-bs $BATCH_SIZE
elif [[ $1 = "resnet34_bot" ]]
then
  $PY_EXE train.py \
-dataset mstar \
-name resnet34_bot \
-hardware "$2" \
-bs $BATCH_SIZE
elif [[ $1 = "resnet50_bot" ]]
then
  $PY_EXE train.py \
-dataset mstar \
-name resnet50_bot \
-hardware "$2" \
-bs $BATCH_SIZE
elif [[ $1 = "resnet101_bot" ]]
then
  $PY_EXE train.py \
-dataset mstar \
-name resnet101_bot \
-hardware "$2" \
-bs $BATCH_SIZE
elif [[ $1 = "resnet152_bot" ]]
then
  $PY_EXE train.py \
-dataset mstar \
-name resnet152_bot \
-hardware "$2" \
-bs $BATCH_SIZE

#######################################################################
##  ResNet (BAT)  #####################################################
#######################################################################

elif [[ $1 = "resnet18_bat" ]]
then
  $PY_EXE train.py \
-dataset mstar \
-name resnet18_bat \
-hardware "$2" \
-bs $BATCH_SIZE
elif [[ $1 = "resnet34_bat" ]]
then
  $PY_EXE train.py \
-dataset mstar \
-name resnet34_bat \
-hardware "$2" \
-bs $BATCH_SIZE
elif [[ $1 = "resnet50_bat" ]]
then
  $PY_EXE train.py \
-dataset mstar \
-name resnet50_bat \
-hardware "$2" \
-bs $BATCH_SIZE
elif [[ $1 = "resnet101_bat" ]]
then
  $PY_EXE train.py \
-dataset mstar \
-name resnet101_bat \
-hardware "$2" \
-bs $BATCH_SIZE
elif [[ $1 = "resnet152_bat" ]]
then
  $PY_EXE train.py \
-dataset mstar \
-name resnet152_bat \
-hardware "$2" \
-bs $BATCH_SIZE

#######################################################################
##  ViT  ##############################################################
#######################################################################

elif [[ $1 = "vit_b_16" ]]
then
  $PY_EXE train.py \
-dataset mstar \
-name vit_b_16 \
-hardware "$2" \
-bs $BATCH_SIZE
elif [[ $1 = "vit_b_32" ]]
then
  $PY_EXE train.py \
-dataset mstar \
-name vit_b_32 \
-hardware "$2" \
-bs $BATCH_SIZE
elif [[ $1 = "vit_l_16" ]]
then
  $PY_EXE train.py \
-dataset mstar \
-name vit_l_16 \
-hardware "$2" \
-bs $BATCH_SIZE
elif [[ $1 = "vit_l_32" ]]
then
  $PY_EXE train.py \
-dataset mstar \
-name vit_l_32 \
-hardware "$2" \
-bs $BATCH_SIZE

#######################################################################
##  MaxViT  ###########################################################
#######################################################################

elif [[ $1 = "max_vit_t" ]]
then
  $PY_EXE train.py \
-dataset mstar \
-name max_vit_t \
-hardware "$2" \
-bs $BATCH_SIZE

#######################################################################
##  NextViT  ###########################################################
#######################################################################

elif [[ $1 = "nextvit_s" ]]
then
  $PY_EXE train.py \
-dataset mstar \
-name nextvit_small \
-hardware "$2" \
-bs $BATCH_SIZE
elif [[ $1 = "nextvit_b" ]]
then
  $PY_EXE train.py \
-dataset mstar \
-name nextvit_base \
-hardware "$2" \
-bs $BATCH_SIZE
elif [[ $1 = "nextvit_l" ]]
then
  $PY_EXE train.py \
-dataset mstar \
-name nextvit_large \
-hardware "$2" \
-bs $BATCH_SIZE

#######################################################################
##  Swin T  ###########################################################
#######################################################################

elif [[ $1 = "swin_t" ]]
then
  $PY_EXE train.py \
-dataset mstar \
-name swin_t \
-hardware "$2" \
-bs $BATCH_SIZE
elif [[ $1 = "swin_s" ]]
then
  $PY_EXE train.py \
-dataset mstar \
-name swin_s \
-hardware "$2" \
-bs $BATCH_SIZE
elif [[ $1 = "swin_b" ]]
then
  $PY_EXE train.py \
-dataset mstar \
-name swin_b \
-hardware "$2" \
-bs $BATCH_SIZE
elif [[ $1 = "swin_v2_t" ]]
then
  $PY_EXE train.py \
-dataset mstar \
-name swin_v2_t \
-hardware "$2" \
-bs $BATCH_SIZE
elif [[ $1 = "swin_v2_s" ]]
then
  $PY_EXE train.py \
-dataset mstar \
-name swin_v2_s \
-hardware "$2" \
-bs $BATCH_SIZE
elif [[ $1 = "swin_v2_b" ]]
then
  $PY_EXE train.py \
-dataset mstar \
-name swin_v2_b \
-hardware "$2" \
-bs $BATCH_SIZE

#######################################################################
##  DeiT  #############################################################
#######################################################################

elif [[ $1 = "deit_tiny_patch16_224" ]]
then
  $PY_EXE train.py \
-dataset mstar \
-name deit_tiny_patch16_224 \
-hardware "$2" \
-bs $BATCH_SIZE
elif [[ $1 = "deit_small_patch16_224" ]]
then
  $PY_EXE train.py \
-dataset mstar \
-name deit_small_patch16_224 \
-hardware "$2" \
-bs $BATCH_SIZE
elif [[ $1 = "deit_base_patch16_224" ]]
then
  $PY_EXE train.py \
-dataset mstar \
-name deit_base_patch16_224 \
-hardware "$2" \
-bs $BATCH_SIZE
elif [[ $1 = "deit_tiny_distilled_patch16_224" ]]
then
  $PY_EXE train.py \
-dataset mstar \
-name deit_tiny_distilled_patch16_224 \
-hardware "$2" \
-bs $BATCH_SIZE
elif [[ $1 = "deit_small_distilled_patch16_224" ]]
then
  $PY_EXE train.py \
-dataset mstar \
-name deit_small_distilled_patch16_224 \
-hardware "$2" \
-bs $BATCH_SIZE
elif [[ $1 = "deit_base_distilled_patch16_224" ]]
then
  $PY_EXE train.py \
-dataset mstar \
-name deit_base_distilled_patch16_224 \
-hardware "$2" \
-bs $BATCH_SIZE

#######################################################################
##  DeiT III  #########################################################
#######################################################################

elif [[ $1 = "deit_tiny_patch16_LS" ]]
then
  $PY_EXE train.py \
-dataset mstar \
-name deit_tiny_patch16_LS \
-hardware "$2" \
-bs $BATCH_SIZE
elif [[ $1 = "deit_small_patch16_LS" ]]
then
  $PY_EXE train.py \
-dataset mstar \
-name deit_small_patch16_LS \
-hardware "$2" \
-bs $BATCH_SIZE
elif [[ $1 = "deit_medium_patch16_LS" ]]
then
  $PY_EXE train.py \
-dataset mstar \
-name deit_medium_patch16_LS \
-hardware "$2" \
-bs $BATCH_SIZE
elif [[ $1 = "deit_base_patch16_LS" ]]
then
  $PY_EXE train.py \
-dataset mstar \
-name deit_base_patch16_LS \
-hardware "$2" \
-bs $BATCH_SIZE
elif [[ $1 = "deit_large_patch16_LS" ]]
then
  $PY_EXE train.py \
-dataset mstar \
-name deit_large_patch16_LS \
-hardware "$2" \
-bs $BATCH_SIZE
elif [[ $1 = "deit_huge_patch14_LS" ]]
then
  $PY_EXE train.py \
-dataset mstar \
-name deit_huge_patch14_LS \
-hardware "$2" \
-bs $BATCH_SIZE

#######################################################################
##  CaiT  #############################################################
#######################################################################

elif [[ $1 = "cait_S24_224" ]]
then
  $PY_EXE train.py \
-dataset mstar \
-name cait_S24_224 \
-hardware "$2" \
-bs $BATCH_SIZE
elif [[ $1 = "cait_XXS24_224" ]]
then
  $PY_EXE train.py \
-dataset mstar \
-name cait_XXS24_224 \
-hardware "$2" \
-bs $BATCH_SIZE
elif [[ $1 = "cait_XXS36_224" ]]
then
  $PY_EXE train.py \
-dataset mstar \
-name cait_XXS36_224 \
-hardware "$2" \
-bs $BATCH_SIZE

#######################################################################
##  ConvNeXt  #########################################################
#######################################################################

elif [[ $1 = "convnext_t" ]]
then
  $PY_EXE train.py \
-dataset mstar \
-name convnext_t \
-hardware "$2" \
-bs $BATCH_SIZE
elif [[ $1 = "convnext_s" ]]
then
  $PY_EXE train.py \
-dataset mstar \
-name convnext_s \
-hardware "$2" \
-bs $BATCH_SIZE
elif [[ $1 = "convnext_b" ]]
then
  $PY_EXE train.py \
-dataset mstar \
-name convnext_b \
-hardware "$2" \
-bs $BATCH_SIZE
elif [[ $1 = "convnext_l" ]]
then
  $PY_EXE train.py \
-dataset mstar \
-name convnext_l \
-hardware "$2" \
-bs $BATCH_SIZE

#######################################################################
##  PatchConv  ########################################################
#######################################################################

elif [[ $1 = "S60" ]]
then
  $PY_EXE train.py \
-dataset mstar \
-name S60 \
-hardware "$2" \
-bs $BATCH_SIZE
elif [[ $1 = "S120" ]]
then
  $PY_EXE train.py \
-dataset mstar \
-name S120 \
-hardware "$2" \
-bs $BATCH_SIZE
elif [[ $1 = "B60" ]]
then
  $PY_EXE train.py \
-dataset mstar \
-name B60 \
-hardware "$2" \
-bs $BATCH_SIZE
elif [[ $1 = "B120" ]]
then
  $PY_EXE train.py \
-dataset mstar \
-name B120 \
-hardware "$2" \
-bs $BATCH_SIZE
elif [[ $1 = "L60" ]]
then
  $PY_EXE train.py \
-dataset mstar \
-name L60 \
-hardware "$2" \
-bs $BATCH_SIZE
elif [[ $1 = "L120" ]]
then
  $PY_EXE train.py \
-dataset mstar \
-name L120 \
-hardware "$2" \
-bs $BATCH_SIZE

else
  echo "No option selected."
fi